from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import bs4
from random_forest_model import y_pred  # fix variable name
from random_forest_model import address


def web_scrapper():
    price_list = []
    # Initialize the web driver
    # make sure to change the path to the actual of the chrome driver
    driver = webdriver.Chrome(executable_path='/path/to/chromedriver')

    # Navigating to the realtor website
    driver.get("https://condos.ca/")

    wanted_address = address  # Remeber to initialize
    # REMEMBER ^^^
    # Searching for the ID of then search bar
    search_bar = driver.find_element_by_xpath(
        "/html/body/div/div/main/div[1]/div[2]/div/div[1]/div/div/div/div/div[2]/div[2]/div/input")
    # Inputting the address we want
    search_bar.send_keys(wanted_address)
    # Finding
    search_bar.send_keys(Keys.RETURN)

    # Wait for the new page to load
    wait = WebDriverWait(driver, 10)
    asking_price_element = wait.until(EC.presence_of_element_located(
        (By.CLASS_NAME, "styles___AskingPrice-sc-54qk44-4")))

    # Get the page source of the new page
    # Essentially records the entirety of the page
    page_source = driver.page_source

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(page_source, 'html.parser')

    # Initialize BeautifulSoup with the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Use the class name to select the specific <div> element
    asking_price_div = soup.find(
        "div", class_="styles___AskingPrice-sc-54qk44-4 deOfjO")

    # Final Line to quite the scraping
    driver.quit()

    for price in asking_price_div:
        if price < y_pred:
            price_list.append(price)

    return price_list
