import nbformat
import sys

notebook_path = "multi_modal_rag_api.ipynb"

def optimize_quota():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            content = cell.source
            
            # 1. Reduce max_workers to 2
            if "with ThreadPoolExecutor(max_workers=5) as executor:" in content:
                content = content.replace("with ThreadPoolExecutor(max_workers=5) as executor:", 
                                         "with ThreadPoolExecutor(max_workers=2) as executor:")
            
            # 2. Add staggering to process_single_chunk
            if "def process_single_chunk(args):" in content and "time.sleep(i * 2)" not in content:
                # Need to ensure time is imported or available
                # Assuming it is since it was used in retry logic
                stagger_code = '    i, chunk, total_chunks = args\n    # Stagger initial requests to avoid thundering herd\n    time.sleep(i % 5 * 2)\n'
                content = content.replace('    i, chunk, total_chunks = args\n', stagger_code)
                
            cell.source = content

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    optimize_quota()
    print("✅ Concurrency optimized for quota limits!")
