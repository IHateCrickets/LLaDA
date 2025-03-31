from html2image import Html2Image
import os
import sys # Added for sys.exit
import shutil # Added for shutil.which

# --- Browser Check ---
def check_browser():
    """Checks if a compatible browser executable is found in PATH."""
    # List of common executable names for Chrome/Edge/Chromium
    browsers = ['chrome', 'msedge', 'chromium', 'chromium-browser']
    for browser in browsers:
        if shutil.which(browser):
            print(f"Found compatible browser: {browser}")
            return True
    return False

if not check_browser():
    print("\nERROR: Compatible browser (Chrome, Edge, or Chromium) not found in system PATH.")
    print("       The html2image library requires one of these browsers to be installed")
    print("       and accessible via the system's PATH environment variable.")
    print("       Please install a compatible browser or ensure its location is added to PATH.")
    sys.exit(1)
# --- End Browser Check ---


# Define the types array to process
types = ['zhihu', 'paper']  # Add all types you need to process here

# Initialize Html2Image object
hti = Html2Image()
# hti.browser.use_new_headless = None # Removed potentially deprecated attribute

for type_txt in types:
    # Ensure png directory exists
    output_dir = os.path.join('png', f"sample_process_{type_txt}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set output path for current type
    hti.output_path = output_dir

    # Loop to generate screenshots
    for i in range(1, 65):
        # Get HTML file path
        html_path = os.path.join('html', f"sample_process_{type_txt}", f'visualization_step_{i}.html')

        if not os.path.exists(html_path):
            print(f"Warning: HTML file not found, skipping step {i} for type '{type_txt}': {html_path}")
            continue

        # Generate and save screenshot
        try:
            print(f"Generating screenshot for step {i}, type '{type_txt}'...")
            hti.screenshot(
                url=html_path,
                save_as=f'visualization_step_{i}.png',
                size=(1200, 500) if type_txt == 'zhihu' else (1200, 800)
            )
        except Exception as e:
            print(f"Error generating screenshot for step {i}, type '{type_txt}': {e}")
            # Optionally decide whether to continue or stop on error
            # continue

print("\nScreenshot generation complete.")
