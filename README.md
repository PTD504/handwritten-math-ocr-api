# Handwritten Math OCR API

---

## Overview

This repository hosts the backend API service responsible for converting handwritten mathematical equations from images into their corresponding LaTeX representations. The API is designed to be consumed by various client applications, such as web-based chatbots or other intelligent systems.

## Features

* **Image to LaTeX Conversion:** Accepts images of handwritten math and returns LaTeX strings.
* **Deep Learning Powered:** Utilizes a robust neural network architecture for high accuracy.
* **Scalable API:** Built with FastAPI for high performance and easy integration.
* **Dockerized Deployment:** Ready for containerized deployment, ensuring consistent environments and simplified scaling.

## Architecture

The service operates as a RESTful API. Upon receiving an image, it performs necessary preprocessing, feeds the image to the trained ML model, and returns the predicted LaTeX string.
