from playwright.sync_api import sync_playwright
import time
import os

def save_article_to_pdf(url, output_pdf):
    with sync_playwright() as p:
        try:
            print("正在启动浏览器...")
            # 启动浏览器
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # 访问网页
            print("正在加载网页...")
            page.goto(url, wait_until="networkidle")
            
            # 等待article元素加载完成
            print("等待内容加载...")
            article = page.wait_for_selector('article')
            
            # 额外等待以确保动态内容加载完成
            time.sleep(2)
            
            # 修改页面样式以优化PDF输出
            page.eval_on_selector('article', '''(article) => {
                // 设置文章样式
                article.style.padding = '20px';
                article.style.margin = '0 auto';
                article.style.maxWidth = '800px';
                
                // 设置图片样式
                const images = article.getElementsByTagName('img');
                for (let img of images) {
                    img.style.maxWidth = '100%';
                    img.style.height = 'auto';
                }
                
                // 设置代码块样式
                const codeBlocks = article.querySelectorAll('pre, code');
                for (let block of codeBlocks) {
                    block.style.whiteSpace = 'pre-wrap';
                    block.style.wordWrap = 'break-word';
                    block.style.overflow = 'auto';
                }
            }''')
            
            # 获取文章内容的边界框
            bbox = article.bounding_box()
            
            print("正在生成PDF...")
            # 将文章内容保存为PDF
            page.pdf(
                path=output_pdf,
                format='A4',
                margin={
                    'top': '1cm',
                    'right': '1cm',
                    'bottom': '1cm',
                    'left': '1cm'
                },
                print_background=True
            )
            
            print(f"PDF已成功保存到: {os.path.abspath(output_pdf)}")
            
        except Exception as e:
            print(f"发生错误: {str(e)}")
        finally:
            browser.close()

if __name__ == "__main__":
    url = "https://www.hyperplasma.top/article/12952/"
    output_pdf = "outputs/article_content.pdf"
    save_article_to_pdf(url, output_pdf)