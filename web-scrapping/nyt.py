import json
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import time
import random
import tqdm

"""
This script scraps NYT by using the NYT api
More information here: https://developer.nytimes.com/apis
"""

# Path to save all downloads to
data_rsis_dir = "../data/RSIS"
get_url_fails = 0
extract_text_fails = 0

options = Options()
options.add_experimental_option('prefs', {
    "download.default_directory" : data_rsis_dir,
    "download.prompt_for_download" : False,
    "plugins.always_open_pdf_externally" : True,
    'profile.managed_default_content_settings.javascript': 2
})
options.add_argument("--disable-javascript")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.maximize_window()

article_list = []
log_list = []

API_KEY = "NYT api key here"

# Getting all links from n pages
print("Getting page urls...")
n = 100
progress_bar = tqdm.tqdm(total=(n - 1) * 10)
for i in range(1, n):
    try:
        
        r = requests.get(f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q=opinion&page={i}&api-key={API_KEY}")
        data = json.loads(r.text)

        # Iterating a total of 10 articles
        for item in data["response"]["docs"]:

            try:
                target_url = item["web_url"]
                progress_bar.set_postfix({
                    "Additional Info": [
                        target_url,
                        get_url_fails,
                        extract_text_fails
                    ]
                })

                # Accessing the webpage
                driver.get(target_url)

                driver.implicitly_wait(5)

                try:
                    # Getitng the title
                    title_element = driver.find_element(By.CSS_SELECTOR, ".css-xkf25q")
                    title = title_element.text
                except:
                    title = ""

                # Getting the body
                # Find the parent element with class "css-53u6y8"
                parent_element = driver.find_elements(By.CLASS_NAME, "css-53u6y8")

                content_list = []
                for parent in parent_element:

                    # Find the desired <p> tags within the parent element
                    p_tags = parent.find_elements(By.CSS_SELECTOR, "p.css-at9mc1.evys1bk0, p.css-12wzsk6.evys1bk0")
                    content_list += [item.text for item in p_tags]
                
                if len(content_list) == 0:
                    extract_text_fails += 1
                    continue
                
                article_data = {
                    "url": target_url,
                    "title": title,
                    "content": content_list
                }

                article_list.append(article_data)

            except Exception as e:
                log_list.append({
                    "extract_index": i,
                    "error": e
                })
                extract_text_fails += 1

            finally:
                progress_bar.update(1)

                # rest randomly
                max_rest_time = 5
                random_number = random.randint(3, max_rest_time)
                time.sleep(random_number)

    except Exception as e:
        log_list.append({
            "url_index": i,
            "error": e
        })
        get_url_fails += 1

log_list.append({
    "url_fails": get_url_fails,
    "extact_text_fails": extract_text_fails
})

with open("data.json", "w") as f:
    json.dump(article_list, f, indent=2)

with open("log.json", "w") as f:
    json.dump(log_list, f, indent=2)
    
print("Done")
