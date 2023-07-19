import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from QaGeneration import QaGeneration

import time
import random
import tqdm

username = "straitstimes_username"
password = "straitstimes_password"

# Log into the website
def login(url:str, username:str, password:str) -> None:
    """
    Logging into straitstimes, users have to manually close ads if there are

    Args:
        url (str): url to log into
        username (str): straitstimes username
        password (str): straitstimes password
    """
    driver.get(url)
    driver.implicitly_wait(10)

    time.sleep(10) # time to close the ad

    driver.find_element(By.ID, "sph_login").click()

    time.sleep(2)

    username_field = driver.find_element(By.NAME, "IDToken1")
    username_field.clear()
    username_field.send_keys(username)

    time.sleep(2)

    password_field = driver.find_element(By.NAME, "IDToken2")
    password_field.clear()
    password_field.send_keys(password)

    time.sleep(2)

    driver.find_element(By.ID, "btnLogin").click()

    time.sleep(15)
    return 

# Get all href for a single url
def get_all_href(url:str, url_class:str, filters:list[str]) -> list:
    """
    Getting all the links from the article

    Args:
        url (str): url to access
        url_class (str): class name for the url
        filters (list[str]): filter the href to ensure the right link is retrieved

    Returns:
        list: list of all the suitable links
    """
    driver.get(url)
    driver.implicitly_wait(10)

    a_tag_list = driver.find_elements(By.CLASS_NAME, url_class)

    href_list = []
    for item in a_tag_list:
        href = item.get_attribute("href")
        if not QaGeneration.check_present(filters, href):
            href_list.append(href)

    return href_list

def get_all_content(url:str) -> dict:
    """
    Getting content from the url

    Args:
        url (str): url to access

    Returns:
        dict: content in a dict form
    """

    driver.get(url)
    driver.implicitly_wait(10)

    title_elem = driver.find_element(By.CSS_SELECTOR, ".headline")
    title = title_elem.text

    paragraphs = driver.find_elements(By.XPATH, "//p[not(@class)]")
    paragraph_list = [para.text + "\n" for para in paragraphs]

    return {
        "url":url,
        "title":title,
        "content":paragraph_list
    }

def save_content(dir:str, filename:str, data:list[dict]) -> None:
    """
    Saves the data to a json file

    Args:
        dir (str): directory for the json file
        filename (str): name of file
        data (list[dict]): data to save
    """
    with open(f"{dir}/{filename}", "w") as f:
        json.dump(data, f, indent=2)

if __name__=="__main__":
    data_dir = "../data/straitstimes"
    get_url_fails = 0
    extract_text_fails = 0

    options = Options()
    options.add_experimental_option('prefs', {
        "download.default_directory" : data_dir,
        "download.prompt_for_download" : False,
        "plugins.always_open_pdf_externally" : True,
    })
    # Javascript disabled
    options.add_argument("--disable-javascript")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.maximize_window()

    filters = [
        "cartoon",
        "forum"
    ]

    home_url = "https://www.straitstimes.com/"

    login(home_url, username, password)
    failure_log_list = []
    
    n = 100
    progress_bar = tqdm.tqdm(total=n)
    href_list = []
    for i in range(n):
        try:
            # getting articles form the opinion section
            url = f"https://www.straitstimes.com/opinion/latest?page={i}"

            progress_bar.set_postfix({
                "Additional Info": [
                    url,
                    get_url_fails,
                    extract_text_fails
                ]
            })
            href_list += get_all_href(url, "stretched-link", filters)
        except Exception as e:
            failure_log_list.append({
                "Error":e
            })
            get_url_fails += 1
        finally:
            # rest randomly
            max_rest_time = 5
            random_number = random.randint(3, max_rest_time)
            time.sleep(random_number)

            progress_bar.update(1)

    progress_bar = tqdm.tqdm(total=len(href_list))
    article_list = []
    for href in href_list:
        try:
            progress_bar.set_postfix({
                "Additional Info": [
                    href,
                    get_url_fails,
                    extract_text_fails
                ]
            })
            article_list.append(get_all_content(href))
        except Exception as e:
            failure_log_list.append({
                "Error":e
            })
            extract_text_fails += 1
        finally:
            # rest randomly
            max_rest_time = 5
            random_number = random.randint(3, max_rest_time)
            time.sleep(random_number)

            progress_bar.update(1)

    save_content(data_dir, "data.json", article_list)

    final_report = {
        "url_fails":get_url_fails,
        "extract_text_fails":extract_text_fails,
        "total_extractions":len(article_list)
    }
    failure_log_list = [final_report] + failure_log_list
    save_content(data_dir, "log.json", failure_log_list)
