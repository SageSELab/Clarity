# Python script to empty a remote directory on bg9 into this one

# This is done because bg9 does not have enough disk space to hold all of the checkpoints, but sciclone does

# make sure you've set up ssh keys between the remote (bg9) and the local (where you run this python script)
# see https://askubuntu.com/questions/46930/how-can-i-set-up-password-less-ssh-login for details

import os
import time

'''while True:
    files = os.popen("ssh ayachnes@bg9.cs.wm.edu ls /scratch/ayachnes/Clarity/neuraltalk2/checkpoint").read().split("\n")
    
    print(files)
    wd = os.listdir(".")
    for f in files:
        if len(f) > 0:
            if not f in wd:
                os.system("scp ayachnes@bg9.cs.wm.edu:/scratch/ayachnes/Clarity/neuraltalk2/checkpoint/"+f+" . ")

    #scp ayachnes@bg9.cs.wm.edu:/scratch/ayachnes/Clarity/neuraltalk2/checkpoint/* .
    time.sleep(360)

'''


import pysftp

WAIT_TIME = 600

host = "bg9.cs.wm.edu"
rdir = "/scratch/ayachnes/Clarity/neuraltalk2/checkpoint/"

with pysftp.Connection(host, username="ayachnes") as sftp:
	sftp.chdir(rdir)

	while True:
		files = sftp.listdir()
		#print(files)
		wd = os.listdir(".")
		for f in files:
			if len(f) > 0:
				if f not in wd:
					start = time.time()
					print("\nGetting " + f + "...")
					sftp.get(f)
					print("Downloaded " + f + " in " + str(int((time.time() - start)*100)/100) + " s")
					sftp.remove(f)
					print("Removed " + host + ":" + rdir + f)

		print("Sleeping " + str(WAIT_TIME) + "s...")
		time.sleep(WAIT_TIME)
