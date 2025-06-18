import re
import json
import os
from pylatexenc.latex2text import LatexNodes2Text
from config import config

def tokenize_latex(formula: str) -> list:
    """
    Tách công thức LaTeX thành danh sách các token.

    Args:
        formula (str): Công thức LaTeX đầu vào.

    Returns:
        list: Danh sách các token (ví dụ: ['x', '^', '2', '+', '1']).
    """
    token_pattern = r'(\\[a-zA-Z]+|[{}_^$%&#]|[0-9]+|[a-zA-Z]+|[^\s])'
    tokens = re.findall(token_pattern, formula)
    return tokens

def load_vocab(filename: str = 'vocab.json') -> tuple:
    """
    Tải vocabulary và ánh xạ ngược từ file vocab.json.

    Args:
        filename (str): Tên file vocab.json (mặc định: 'vocab.json').

    Returns:
        tuple: (vocab, idx2char)
            - vocab (dict): Ánh xạ từ token sang ID (e.g., {'x': 0, '+': 1, ...}).
            - idx2char (dict): Ánh xạ từ ID sang token (e.g., {0: 'x', 1: '+', ...}).
    """
    with open(f'{config.model_dir}/{filename}', 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = data['vocab']
    idx2char = {int(k): v for k, v in data['idx2char'].items()}
    return vocab, idx2char

def tokens_to_latex(token_ids: list, idx2char: dict) -> str:
    """
    Chuyển danh sách token ID thành công thức LaTeX.

    Args:
        token_ids (list): Danh sách các ID của token.
        idx2char (dict): Ánh xạ từ ID sang token.

    Returns:
        str: Công thức LaTeX được xây dựng từ token_ids.
    """
    # Loại bỏ các token đặc biệt (<sos>, <eos>, <pad>)
    filtered_ids = [tid for tid in token_ids if tid in idx2char and idx2char[tid] not in [config.sos_token, config.eos_token, config.pad_token]]
    # Chuyển ID thành token và nối thành chuỗi
    latex = ' '.join([idx2char[tid] for tid in filtered_ids])
    return latex

def latex_validator(latex: str) -> tuple:
    """
    Kiểm tra và sửa lỗi mã LaTeX để đảm bảo có thể render được.

    Args:
        latex (str): Mã LaTeX cần kiểm tra.

    Returns:
        tuple: (bool, str)
            - bool: True nếu LaTeX hợp lệ, False nếu không.
            - str: Mã LaTeX đã sửa hoặc thông báo lỗi.
    """
    corrected_latex = latex.strip()

    # 1. Kiểm tra và thêm dấu $ nếu cần (chỉ cho inline math)
    if not re.search(r'\$.*\$', corrected_latex) and not re.search(r'\\\[.*\\\]', corrected_latex) and not re.search(r'\\\(.*\\\)', corrected_latex):
        corrected_latex = f'${corrected_latex}$'

    # 2. Sửa các lỗi phổ biến
    # Thêm dấu {} cho \frac, \sqrt, ^, _
    corrected_latex = re.sub(r'\\frac\s*([^{])', r'\\frac{\1}', corrected_latex)
    corrected_latex = re.sub(r'\\sqrt\s*([^{])', r'\\sqrt{\1}', corrected_latex)
    corrected_latex = re.sub(r'\^([^{])', r'^{\1}', corrected_latex)
    corrected_latex = re.sub(r'_([^{])', r'_{\1}', corrected_latex)
    # Xóa các lệnh rỗng
    corrected_latex = re.sub(r'\\\w+\s*{\s*}', '', corrected_latex)
    # Sửa \frac thiếu tham số
    corrected_latex = re.sub(r'\\frac{[^}]*}{}', r'\\frac{1}{1}', corrected_latex)  # Thay thế \frac{a}{} thành \frac{a}{1}

    # 3. Xử lý dấu ngoặc không khớp
    stack = []
    output = []
    for char in corrected_latex:
        if char == '{':
            stack.append(char)
            output.append(char)
        elif char == '}':
            if stack and stack[-1] == '{':
                stack.pop()
                output.append(char)
            else:
                output.append(char)  # Giữ nguyên nhưng ghi log
        else:
            output.append(char)
    
    # Thêm dấu } nếu thiếu
    for _ in stack:
        output.append('}')
    
    corrected_latex = ''.join(output)

    # 4. Kiểm tra bằng pylatexenc
    try:
        LatexNodes2Text().latex_to_text(corrected_latex)
        return True, corrected_latex
    except Exception as e:
        return False, r"\text{Không thể trích xuất công thức từ ảnh.}"