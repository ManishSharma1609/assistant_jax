import os
import re
import json
from bs4 import BeautifulSoup
from boilerpy3 import extractors
from tqdm import tqdm
import hashlib

class JAXDocsPreprocessor:
    def __init__(self, input_dir='jax-docs1', output_dir='processed_jax_docs'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.extractor = extractors.ArticleExtractor()
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_html(self, soup):
        """Remove unnecessary elements while preserving code blocks"""
        # Remove navigation and decorative elements
        for element in soup(['header', 'footer', 'nav', 'script', 'style', 
                           'iframe', 'img', 'svg', 'button', 'form']):
            element.decompose()
        
        # Remove divs with common classes used for navigation
        for div in soup.find_all('div', class_=re.compile(r'sidebar|toc|navigation|nav')):
            div.decompose()
        
        return soup
    
    def extract_code_blocks(self, soup):
        """Extract and preserve code examples"""
        code_blocks = []
        for pre in soup.find_all('pre'):
            code = pre.get_text().strip()
            if len(code.split('\n')) > 1 or len(code) > 50:  # Only keep substantial code blocks
                code_blocks.append(code)
                # Replace with a marker we can reference later
                marker = f"CODE_BLOCK_{len(code_blocks)-1}"
                pre.replace_with(f"[{marker}]")
        
        return code_blocks
    
    def process_content(self, text, code_blocks):
        """Process content and reintegrate code blocks"""
        # Normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Reinsert code blocks with markers
        for i, code in enumerate(code_blocks):
            text = text.replace(f"[CODE_BLOCK_{i}]", f"\n```python\n{code}\n```\n")
        
        return text
    
    def process_file(self, file_path):
        """Process a single documentation file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        # Extract and clean
        soup = self.clean_html(soup)
        code_blocks = self.extract_code_blocks(soup)
        title = soup.title.string if soup.title else os.path.basename(file_path)
        content = soup.get_text(separator='\n', strip=True)
        processed_content = self.process_content(content, code_blocks)
        
        # Create document ID
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        
        return {
            'id': doc_id,
            'title': title,
            'path': file_path,
            'content': processed_content,
            'code_blocks': code_blocks,
            'source': 'jax-docs'
        }
    
    def run(self):
        """Process all HTML files and create a knowledge base"""
        html_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.html'):
                    html_files.append(os.path.join(root, file))
        
        knowledge_base = []
        for file_path in tqdm(html_files, desc="Processing JAX docs"):
            try:
                doc = self.process_file(file_path)
                knowledge_base.append(doc)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Save knowledge base
        output_path = os.path.join(self.output_dir, 'jax_knowledge_base.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2)
        
        print(f"Processed {len(knowledge_base)} files. Knowledge base saved to {output_path}")
        return knowledge_base

if __name__ == "__main__":
    preprocessor = JAXDocsPreprocessor()
    knowledge_base = preprocessor.run()