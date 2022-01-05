'''

 Script to automate getting all files from a specific directory on a remote host.
 
 Note 1: make sure you've set up ssh keys between the remote 
 (where you're fetching files from) and the local 
 (where you run this python script, where the files are transferred to)
 
 Note 2: make sure that the pysftp is installed on the machine on which
 you use this script; `pip install pysftp`
 
 
 
 Setting up ssh keys:
 
 # I want to ssh from machine A to machine B without having to enter the password on machine A every time

 # On machine A, execute the following:
 # (also, use an empty password)

 ssh-keygen


 # Then run the following, replacing user@host with machine B's user and IP:

 ssh-copy-id user@host

 #  (or if your server uses custom port number):
 ssh-copy-id -p 6174 user@host


 # Congratulations, you can now ssh or sftp from machine A to machine B without entering a password on machine A
 
 '''

import os
import time
import sys
import pysftp


if len(sys.argv) < 3:
    print("\nScript to automate getting all files from a specific directory on a remote host,")
    print("outputting fetched files to the working directory\n")
    print("python " + __file__ + " <user@host> <remote_directory>") 
    print("i.e. python " + __file__ + " ayachnes@bg9.cs.wm.edu /scratch/ayachnes/Clarity/neuraltalk2/checkpoint/") 
    print("Note: See the script's source for dependency setup\n")
    exit()

if sys.argv[1].find("@") == -1:
    print("Error: first argument must be of the form `user@host`")
    exit()

user = sys.argv[1].split("@")[0]
host = sys.argv[1].split("@")[1]
rdir = sys.argv[2]

if len(user) == 0:
    print("Error: empty user")
    exit()
    
if len(host) == 0:
    print("Error: empty host")
    exit()

# connect to the remote host, cd to the given directory, and fetch all of the files
# from the remote host to the working directory

with pysftp.Connection(host, username=user) as sftp:
    sftp.chdir(rdir)

    files = sftp.listdir()
    num_files = len(files)
    num_files_fetched = 0

    for f in files:
        if len(f) > 0:
            start = time.time()
            sys.stdout.write("Getting " + f + "... (" + str(int((float(num_files_fetched)/num_files)*100)) + "%)\n")
            sys.stdout.flush()
            sftp.get(f)
            sys.stdout.write("Downloaded " + f + " in " + str(int((time.time() - start)*100)/100) + " s\n")
            sys.stdout.flush()
            num_files_fetched += 1

