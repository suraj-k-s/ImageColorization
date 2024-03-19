# Image Colorization Processing Web Application

This Django web application allows users to process Image Colorization by uploading image . The processed images displayed .

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/suraj-k-s/ImageColorization.git
    ```

2. Navigate to the project directory:

    ```bash
    cd ImageColorization
    ```
    
3. Installing Virtualenv 
    
    ```
    pip install virtualenv
    ```
    
4. Creating Virtualenv
    ```
    virtualenv omrenv
    ```

    OR

   ```
    python -m venv omrenv
    ```
    
5. Activating Environment:
    ```
    omrenv\Scripts\activate
    ```

6. Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Django development server:

    ```bash
    python manage.py runserver
    ```

2. Open a web browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to access the application.

3. Upload Image.

4. Click the "Process Image" button to process the uploaded image to Colorized Image.

5. After processing, the image will be show in page.


## License

This project is licensed under the [MIT License](LICENSE).
