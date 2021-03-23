from bs4 import BeautifulSoup
import requests
import os


image_path = 'https://bulbapedia.bulbagarden.net/wiki/List_of_Pok%C3%A9mon_by_base_stats_(Generation_VI)'
download_dir = 'pokemon'

response = requests.get(f'{image_path}')
soup = BeautifulSoup(response.text,'lxml')

#print(soup)

results = soup.find_all("img", limit= 798)

image_links = [result.get('src') for result in results]
print(image_links)

for index, link in enumerate(image_links):
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    
    img = requests.get('https:'+link)
    print(img)
    with open(download_dir+'\\'+ str(index+1)+'.jpg','wb') as file:
        file.write(img.content)
#print(results)
