import sys
import requests

current_filename = sys.argv[1]
current_ip = sys.argv[2]
gpu_num = sys.argv[3]
target_url = 'https://maker.ifttt.com/trigger/finish_alarm/with/key/d9RsogUIiHCtyrDGKDUu7Q'
requests.post(target_url, data={'value1': current_filename, 'value2': current_ip, 'value3': gpu_num})