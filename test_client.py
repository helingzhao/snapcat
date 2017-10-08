import requests

url = 'http://127.0.0.1:5000/'
files = {
   'file': open('testCat.jpg', 'rb')
}
requests.post(url, files=files)

#from server import load_categories, PATH_TO_LABELS

#print(load_categories(PATH_TO_LABELS))
