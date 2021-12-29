from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from PIL import Image
import io
import aiohttp
import asyncio
from progress_bar import ProgressBar

class ImageDownloader:
    def __init__(self, webdriver_path):
        self.webdriver_path = webdriver_path
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('headless')
        self.driver = webdriver.Chrome(self.webdriver_path, options=self.options)
        self.urls = set()
    
    async def download_task(self, url_data):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url, path, filename in url_data:
                task = asyncio.ensure_future(self.retrieve_image(session, url, path, filename))
                tasks.append(task)

            files = await asyncio.gather(*tasks)
            for image_file, path, filename, url in files:
                try:
                    if image_file == None:
                        continue
                    image = Image.open(image_file)
                    with open(f'{path}/{filename}', 'wb') as f:
                        try:
                            image.save(f, 'JPEG')
                        except:
                            image.save(f, 'PNG')
                except Exception as e:
                    print(f'DOWNLOAD FAILED for {filename}- {e}')
                    self.urls.remove(url)
    
    async def retrieve_image(self, session, url, path, filename):
        try:
            async with session.get(url, timeout=5) as response:
                image_file = io.BytesIO(await response.content.read())
                return (image_file, path, filename, url)
        except:
            print(f'TIMEOUT - {filename}')
            self.urls.remove(url)
            return (None,)*4

    def download_images(self, query, limit, path):
        self.driver.get('https:images.google.com')
        search_bar = self.driver.find_element_by_name('q')
        search_bar.send_keys(query)
        search_bar.send_keys(Keys.RETURN)
        self.urls = set()
        pb = ProgressBar(max_value=limit)

        while len(self.urls) < limit:
            thumbnails = self.driver.find_elements(By.CLASS_NAME, 'Q4LuWd')
            for thumbnail in thumbnails:
                try:
                    thumbnail.click()
                except:
                    pass
                expanded = self.driver.find_elements(By.CLASS_NAME, 'n3VNCb')
                for image in expanded:
                    if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                        old_len = len(self.urls)
                        self.urls.add(image.get_attribute('src'))
                        if old_len != len(self.urls):
                            asyncio.run(self.download_task([(image.get_attribute('src'), path, f'{query}{len(self.urls)}.jpg')]))
                        print(f'Download progress: {pb.update(len(self.urls)-old_len)} {len(self.urls)}/{limit} done', end='\r')
                if len(self.urls) >= limit:
                    break

        self.driver.quit()

downloader = ImageDownloader('C:\Program Files (x86)\\chromedriver.exe')
downloader.download_images('mazda car', 500, 'images/mazda')