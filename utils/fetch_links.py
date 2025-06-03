from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
import weasyprint  # 导入WeasyPrint库

# 创建输出文件夹
output_folder = 'outputs/merck'
os.makedirs(output_folder, exist_ok=True)

# 设置Selenium的Chrome选项
chrome_options = Options()
chrome_options.add_argument("--headless")  # 启动无头模式，不显示浏览器界面
chrome_options.add_argument("--disable-gpu")  # 禁用GPU加速

# 启动Chrome浏览器
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# 设置基础URL和搜索页URL
base_url = "https://www.merckvetmanual.com"
search_url_template = "https://www.merckvetmanual.com/searchresults?query=dog&page={}"

# 存储所有文章链接
article_links = []

# 爬取从 page=1 到 page=10 的搜索结果
for page in range(1, 11):
    search_url = search_url_template.format(page)
    print(f"Fetching page {page}: {search_url}")

    # 打开网页
    driver.get(search_url)

    # 等待页面加载
    time.sleep(3)  # 等待3秒，确保页面加载完成

    # 获取页面内容
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 查找所有文章链接
    search_results = soup.find_all('a', class_='AllSearchItems_title__9c_2j')
    
    for link in search_results:
        href = link.get('href')
        if href:
            full_url = urljoin(base_url, href)  # 拼接完整的URL
            article_links.append(full_url)

# 关闭浏览器
driver.quit()

# 下载每个链接的页面并保存为PDF
for article in article_links:
    try:
        print(f"Downloading {article}")
        # 使用 WeasyPrint 下载并保存为 PDF
        weasyprint.HTML(article).write_pdf(os.path.join(output_folder, f"{article.split('/')[-1]}.pdf"))
    except Exception as e:
        print(f"Failed to download {article}: {e}")

print(f"Downloaded {len(article_links)} PDFs to '{output_folder}'.")
