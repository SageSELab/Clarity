
math.randomseed(os.time())

function print(s)

end

require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

local time_fmt = tostring(os.time()) --added by Ali; this is for giving the .t7 files a unique id (id is the time the training started)
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5','coco/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','coco/data.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam') 
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', './checkpoint', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', time_fmt, 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123 --[[math.random(99999999)]] , 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-preprotype', 'both', 'the preprocessing type that was done on data; either high, low, or both (defaults to both)')
cmd:option('-csv_out', "", "an output csv file for the bleu scores, used during hyperparameter search"); --it is assumed that csv_out ~= "" implies that it is a hyperparameter search (no saving of checkpoints)
cmd:option('-beam_size', 2, "beam size used for sampling")
cmd:option('-save_cp', 1, 'whether to save .t7 and .json checkpoints; 0 = dont save, 1 = save')

cmd:text()

local csv_wrote_bleu_score = false --variable that is only relevant during a hyperparameter search


-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)

--dump opt to stdout

io.write("\n\ntrain.lua was run with the following command line arguments: \n\n")

for k,v in pairs(opt) do
	io.write(k, ": ", v, "\n")
end

io.write("\n\n")

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
else
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = 1
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  protos.lm = nn.LanguageModel(lmOpt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)
  protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
io.write('total number of parameters in LM: ', params:nElement() .. "\n")
io.write('total number of parameters in CNN: ', cnn_params:nElement() .. "\n")
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()

collectgarbage() -- "yeah, sure why not"
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(iter, split, evalopt)

  local function tabletostring(tab)
    local str = "{"
        for key, val in pairs(tab) do
            str = str .. key .. " = " .. val .. ", "
        end
    str = str:sub(1, #str - 2) .. "}"
    
    return str
  end
  
  --io.write("eval_split called with iter = " .. tostring(iter) .. ", split = " .. tostring(split) .. ", evalopt = " .. tabletostring(evalopt) .."\n")
  
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  --io.write("val_images_use = " .. tostring(val_images_use))

  protos.cnn:evaluate()
  protos.lm:evaluate()
  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local predictions = {}
  
  local vocab = loader:getVocab()
  
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    
    -- evaluate loss if we have the labels
    local loss = 0
    
    if data.labels then
      local expanded_feats = protos.expander:forward(feats)
      local logprobs = protos.lm:forward{expanded_feats, data.labels}
      loss = protos.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss
      loss_evals = loss_evals + 1
    end
    

    --NOTE: the following is from eval.lua, replacing the commented out code after it
    
    
    
    --[[cmd:option('-sample_max', 10, '1 = sample argmax words. 0 = sample from distributions.')
    cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')]]
    
    
    local sample_opts = {sample_max = 0, beam_size = opt.beam_size, temperature = 1.0}
    local seq = protos.lm:sample(feats, sample_opts)

    -- forward the model to also get generated samples for each image
    --local seq = protos.lm:sample(feats)
    

    
    local sents = net_utils.decode_sequence(vocab, seq)

    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, caption = sents[k], path = data.infos[k].file_path}
      


     --[[ io.write("entry " .. tostring(k) .. " {")
      
      io.write("image_id = " .. tostring(entry.image_id) .. ", ")

      for key, val in pairs(entry) do
         io.write(key .. ' = ' .. tostring(val) .. ", ")
      end

      io.write("}\n\n")]]

      table.insert(predictions, entry)
      if verbose then
        if k==1 then print(entry.caption) end --take a look at caption
        --print(string.format('image %s: %s', entry.path, entry.caption))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss))
    end
    
    

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(iter, predictions, opt.id, opt.preprotype)
	--[[for im, val in pairs(predictions) do
		--io.write(val.path, val.caption, "\n")
        io.write(val.caption, "\n")
    end]]
  end

  return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224 
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  -- we have to expand out image features, once for each sentence
  local expanded_feats = protos.expander:forward(feats)
  -- forward the language model
  local logprobs = protos.lm:forward{expanded_feats, data.labels}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.labels)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop language model
  local dexpanded_feats, ddummy = unpack(protos.lm:backward({expanded_feats, data.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    local dfeats = protos.expander:backward(feats, dexpanded_feats)
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score = nil
while true do  

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    io.write(string.format('\n\n\n************************** Iteration %d      total_loss: %f **************************\n\n\n', iter, losses.total_loss))
    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats = eval_split(iter, 'val', {val_images_use = opt.val_images_use})
    io.write('validation loss: ', val_loss, "\n")
    
    

    
    
    val_loss_history[iter] = val_loss
    if lang_stats then
    
        io.write("\n\n")
        for key, val in pairs(lang_stats) do
            io.write(key .. " = " .. tostring(val) .. "\n")
        end
        io.write("\n\n")
    
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = path.join(opt.checkpoint_path, 'model-' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_lang_stats_history = val_lang_stats_history

    if string.len(opt.csv_out) == 0 and opt.save_cp == 1 then --if we aren't doing a hyperparameter search AND cp saving is enabled
        utils.write_json(checkpoint_path .. '.json', checkpoint)
        io.write('\nwrote json checkpoint to ' .. checkpoint_path .. '.json\n')
    end
    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use average BLEU score for deciding how well we did
      current_score = (lang_stats['Bleu_1'] + lang_stats['Bleu_2'] + lang_stats['Bleu_3'] + lang_stats['Bleu_4'])/4
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    
    if string.len(opt.csv_out) == 0 and opt.save_cp == 1 then --if we aren't doing a hyperparameter search AND cp saving is enabled
    
        if (best_score == nil or current_score > best_score) then
           if iter > 0 then --don't save on very first write
             best_score = current_score
             -- include the protos (which have weights) and save to file
             local save_protos = {}
             save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
             save_protos.cnn = thin_cnn
             checkpoint.protos = save_protos
             -- also include the vocabulary mapping so that we can use the checkpoint 
             -- alone to run on arbitrary images without the data loader
             checkpoint.vocab = loader:getVocab()
             torch.save(checkpoint_path .. '.t7', checkpoint)
             io.write('wrote checkpoint to ' .. checkpoint_path .. '.t7\n\n') 
             io.write("current_score = best_score = " .. tostring(best_score) .. "\n")
            end
        else
            io.write("Did not write checkpoint because best_score = " .. tostring(best_score) .. " > current_score = " .. tostring(current_score) .. "\n")
            --io.write("iter = " .. tostring(iter) .. "\n")
        end
    elseif string.len(opt.csv_out) > 0 then --we're doing a hyperparameter search, so just write something to csv
        
        io.write("current_score = " .. tostring(current_score))
        
        local csv_file = assert(io.open(opt.csv_out, "a")) --open the csv file in append mode
        
        if (iter == 0) then --first iteration? Dump the opt table
        
        
            local line = "";
            
            for k,v in pairs(opt) do
                line = line .. "NEW SESSION,"
            end
            
            line = line:sub(1, #line - 1) .. "\n" --get rid of the last comma and add a newline

            csv_file:write(line)




            line = "";
            
            for k,v in pairs(opt) do
                line = line .. k .. ","
            end
            
            line = line:sub(1, #line - 1) .. "\n" --get rid of the last comma and add a newline

            csv_file:write(line)
            
            
            
            
            line = "";
            
            for k,v in pairs(opt) do
                line = line .. v .. ","
            end
            
            line = line:sub(1, #line - 1) .. "\n" --get rid of the last comma and add a newline
            
            csv_file:write(line)
            
        
        elseif iter == (opt.max_iters - 1) then --we are done with this session, so add whitespace for the next session
        
        
            for i = 1, 4 do --spacing for bleu score block
                csv_file:write(" , , , , , , , \n")
            end
        
            for i = 1, 10 do

                local line = "";
                
                for k,v in pairs(opt) do
                    line = line .. " ,"
                end
                
                line = line:sub(1, #line - 1) .. "\n" --get rid of the last comma and add a newline

                csv_file:write(line)

            end
            
        else
        
            if not csv_wrote_bleu_score then --if we have never written a bleu score to the csv before (i.e. this is the first time reaching this if block since iteration 0)
                csv_wrote_bleu_score = true --ensure we never execute these lines again
                
                for i = 1, 4 do --spacing for bleu score block
                    csv_file:write(" , , , , , , , \n")
                end
                
                csv_file:write("iteration, CIDEr, BLEU-1, BLEU-2, BLEU-3, BLEU-4, BLEU-AVG, ROUGE_L, METEOR, VAL_LOSS\n") --write the appropriate headers to the csv
                
            end
            
            local avg = (lang_stats['Bleu_1'] + lang_stats['Bleu_2'] + lang_stats['Bleu_3'] + lang_stats['Bleu_4'])/4
            
            local line = tostring(iter) .. ", " .. tostring(lang_stats['CIDEr']) .. ", " .. tostring(lang_stats['Bleu_1']) .. ", "  .. tostring(lang_stats['Bleu_2']) .. ", " 
            .. tostring(lang_stats['Bleu_3']) .. ", " .. tostring(lang_stats['Bleu_4']) .. ", " .. tostring(avg) .. ", " .. tostring(lang_stats['ROUGE_L']) .. ", " 
            .. tostring(lang_stats['METEOR']) .. ", " .. tostring(val_loss) .. "\n";
                
            csv_file:write(line) --populate the csv with the scores of this iteration
            
        end

        io.close(csv_file)
        
    end
    io.write("\n**********************************************************************************************\n\n")
    
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    io.write('loss seems to be exploding, quitting.\n')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
  iter = iter + 1
end
