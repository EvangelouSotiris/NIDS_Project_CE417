#!/bin/bash
sudo go run capture.go $(echo -e "import netifaces\nprint(netifaces.gateways()['default'][netifaces.AF_INET][1])" | python3) | sudo python writefile.py | sudo python outputfields.py | python main.py

# sudo env "PATH=$PATH"
