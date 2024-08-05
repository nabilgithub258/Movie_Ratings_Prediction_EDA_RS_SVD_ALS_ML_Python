# Movie Ratings Prediction

## Overview
This project focuses on predicting movie ratings using collaborative filtering techniques. We analyzed a movie dataset and employed various models to enhance the prediction accuracy of user ratings.

## Data Preparation
1. **Data Merging**: We merged different datasets using the `merge` function with the key being `title_id` to combine movie information and user ratings.
2. **Initial Recommendation System**: We created a simple recommendation system based on the `corrwith` function to find correlations between movie ratings.

## Methodology
1. **Predicted Matrix**: We constructed a predicted ratings matrix to evaluate the initial performance.
2. **Train-Test Split**: We split the data into training and testing sets to validate our models properly.

### Singular Value Decomposition (SVD)
- **Implementation**: We used SVD for our first model.
- **Performance**: The SVD model achieved an RMSE of 0.69.

### Alternating Least Squares (ALS)
- **Implementation**: We then implemented the ALS model for better performance.
- **Performance**: The ALS model further improved the accuracy, resulting in our best RMSE of 0.50.

## Results
- **SVD**: RMSE = 0.69
- **ALS**: RMSE = 0.50

## Conclusion
The ALS model provided significant improvements over SVD, demonstrating its effectiveness in movie ratings prediction.

## Dependencies
**pandas**
**numpy**
**scikit-learn**
**scipy**
**implicit**

## Contributing
We welcome contributions to this project! If you have any suggestions, bug reports, or improvements, please follow these steps:

**Fork the repository.**
**Create a new branch for your feature or bugfix.**
**Make your changes and commit them with clear and descriptive messages.**
**Push your changes to your forked repository.**
**Submit a pull request to the main repository.**

## Acknowledgements
We would like to acknowledge the sources and libraries that made this project possible.

## Contact
For any questions or inquiries, please contact us at [nabilmomin1989@gmail.com].
