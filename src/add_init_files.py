import os

EXCLUDE_DIRS = {}

def should_skip(path):
    if path == ".":
        return False
    parts = path.split(os.sep)
    return any(part.startswith('.') or part in EXCLUDE_DIRS for part in parts)

def contains_python_files_anywhere(dirpath):
    for root, _, files in os.walk(dirpath):
        if any(f.endswith(".py") for f in files):
            return True
    return False

def add_init_files(root_dir):
    root_dir = os.path.abspath(root_dir)
    print(f"🔍 Walking: {root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        print(f"\n📁 Checking: {rel_path}")

        if should_skip(rel_path):
            print(f"⏩ Skipped: {rel_path}")
            dirnames[:] = []
            continue

        if not contains_python_files_anywhere(dirpath):
            print(f"❌ No .py files under: {rel_path}")
            continue

        init_path = os.path.join(dirpath, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                pass
            print(f"✅ Added: {init_path}")
        else:
            print(f"🔹 Already exists: {init_path}")

if __name__ == "__main__":
    add_init_files(".")
