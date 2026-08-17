from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time


url = "https://www.amazon.in/product-reviews/B088BGY43C"

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

try:
    print("Opening Amazon...")
    driver.get(url)

    time.sleep(8)

    print("\nPage title:")
    print(driver.title)

    print("\nCurrent URL:")
    print(driver.current_url)

    print("\nPage source length:")
    print(len(driver.page_source))

    # Save the page we actually received
    with open("output/amazon_page.html", "w", encoding="utf-8") as file:
        file.write(driver.page_source)

    print("\nPage saved to:")
    print("output/amazon_page.html")

    input("\nPress Enter to close Chrome...")

finally:
    driver.quit()