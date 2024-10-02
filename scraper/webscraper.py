import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os


# get all the chapter URL from the main URL
def get_chapter_URL(URL):
    chapter_list = []
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    div = soup.find_all('div', class_ = 'bg-bg-secondary p-3 rounded mb-3 shadow')
    for chapter in div:
        site = chapter.find('a', class_ = 'text-sm md:text-base font-bold mb-1 text-text-normal').attrs['href']
        chapter_list.append(site)
    return chapter_list

# get all the images URL from the chapter URL
def get_images_url(chapter_list, test=False):
    colour_images_url = []
    # if test is True, only download the first chapter
    if test:
        chapter_list = chapter_list[:1]
    for i in tqdm(range(len(chapter_list)), "getting colour images"):
        page_images = []
        html_text = requests.get(chapter_list[i], 'r').text
        soup = BeautifulSoup(html_text, 'html.parser')
        all_imgs = soup.find_all('div', class_ = 'text-center')
        for img in all_imgs:
            site = img.find("img", class_ = 'mb-3 mx-auto js-page')
            if site is not None:
                page_images.append(site['src'])
        colour_images_url = colour_images_url + page_images
    # remove the first 3 and last 3 images as they are not part of the manga
    colour_images_url = colour_images_url[3:-3]
    return colour_images_url


# download the images from the images URL
def download_images(images_URL, location, test=False):
    output_folder = location
    os.makedirs(output_folder, exist_ok=True)
    # if test is True, only download the last 10 images
    if test:
        images_URL = images_URL[-10:]
    existing_files = os.listdir(output_folder)
    num_existing_files = len(existing_files)
    for i, url in enumerate(tqdm(images_URL, desc='Downloading images')):
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Generate a filename based on the index (i)
            filename = os.path.join(output_folder, f'{num_existing_files + 1}.jpg')
            num_existing_files += 1
            # Save the image to the specified filename
            with open(filename, 'wb') as file:
                file.write(response.content)

            tqdm.write(f"Downloaded: {url} => {filename}")
        except Exception as e:
            tqdm.write(f"Error downloading {url}: {str(e)}")

# all in one function to get the data
def get_data(URL, location, maintest=False):
    URL_list = get_chapter_URL(URL)
    images_URL = get_images_url(URL_list, test=maintest)
    download_images(images_URL, location, test=maintest)

# demonslayer = "https://ww7.demonslayermanga.com/manga/kimetsu-no-yaiba-digital-colored-comics/"
# get_data(demonslayer, "/dcs/large/u2146727/colour")

# haikyuu = "https://ww8.readhaikyuu.com/manga/haikyu-digital-colored-comics/"
# get_data(haikyuu, "/dcs/21/u2146727/cs310/dataset/colour", maintest=True)

# kaguya = "https://ww3.readkaguyasama.com/manga/kaguya-sama-love-is-war-digital-colored-comics/"
# get_data(kaguya, "/dcs/large/u2146727/colour")

#1320
    
demonslayer = "https://ww7.demonslayermanga.com/manga/demon-slayer-kimetsu-no-yaiba"
# CHANGE DIRECTORY
get_data(demonslayer, "/dcs/large/u2146727/original", maintest=False)