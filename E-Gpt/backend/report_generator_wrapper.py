# report_generator_wrapper.py
# Wrapper to run the report generator with proper path handling

def run_report_generator():
    import sys
    import os
    import runpy
    
    # Add the report generator directory to the Python path
    report_generator_path = os.path.join(os.path.dirname(__file__), "report generator")
    if report_generator_path not in sys.path:
        sys.path.insert(0, report_generator_path)
    
    # Run the report generator app
    runpy.run_module('ui.app', run_name='__main__')

if __name__ == "__main__":
    run_report_generator()