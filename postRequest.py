# POSTMAN PROGRAMINA ALTERNATİF OLARAK POST YAPILMASI İÇİN TOOL
import requests  # Http request göndermemize yarayan bilindik modül
url = 'http://127.0.0.1:8080/boxVsbag' # Localhost'ta çalışan flask uygulamamızın url'si
myobj = {'base64_string': base64String}  # Post edeceğimiz data

x = requests.post(url, json = myobj)  # İstek gönderiliyor

print(x.text)  # İsteğimize karşılık dönen yanıt