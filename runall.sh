#!/bin/bash
sudo go run capture.go enp3s0 | sudo python writefile.py | sudo python outputfields.py | python main.py

# sudo env "PATH=$PATH"
