# JamboTree-LinearRegression
<p align="center">
  <img width="100%" alt="image" src="https://media.licdn.com/dms/image/v2/D5612AQE1XxEXI1JDdQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1691710880639?e=2147483647&v=beta&t=SDtiydTUX3t-5LrHMgrVKZ-zVaddKaG3cSxRRDJhThc">
</p>


## 🏷️Overview


The objective of conducting linear regression analysis for Jamboree is to identify the significant factors influencing graduate admissions, such as GRE scores,
 CGPA, and Letters of Recommendation (LOR), while exploring the interrelationships among these features and assessing any multicollinearity issues. By
 developing a predictive model to estimate admission probabilities based on the identified features, we aim to provide prospective students with valuable
 insights into their chances of admission. Additionally, we will evaluate the model's assumptions, particularly regarding the normality and homoscedasticity of
 residuals, to ensure its robustness. Ultimately, this analysis will empower Jamboree to enhance its advising strategies, helping students focus on improving
 their performance in critical areas for better admission outcomes.

 ## 📚 Dataset
1.   Serial No. (Unique row ID)
2.   GRE Scores (out of 340)
3.   TOEFL Scores (out of 120)
4.   University Rating (out of 5)
5.   Statement of Purpose and Letter of Recommendation Strength (out of 5)
6.   Undergraduate GPA (out of 10)
7.   Research Experience (either 0 or 1)
8.   Chance of Admit (ranging from 0 to 1)


## Key Insight

- Upon conducting regression analysis, it's evident that **CGPA** emerges as the **most influential feature** in 
predicting admission chance.- Additionally, **GRE and TOEFL** scores also exhibit significant importance in the predictive model
- Here’s a concise bullet-point summary :
    - Initial regression model through **OLS** revealed **University Rating** and **SOP** as non-relevant features.
    -  **Multicollinearity Check**:- VIF scores consistently below **5**, indicating low multicollinearity among predictors.
    -  **Residual Analysis**:- **Residuals do not follow a normal distribution**.
    -  Presence of **Homoscedasticity** in residual plots.
    -   **Regularized Models**:
        - **Ridge and Lasso regression** results were comparable to the Linear Regression Model.
    - Overall, the features demonstrated strong predictive capabilities.
      
## Recommendations
 -  Feature Enhancement:
    -  Focus Areas: Encourage students to prioritize improving their GRE scores,Cumulative Grade Point Average (CGPA),TOE
 FL and the quality of Letters of Recommendation (LOR). These three factors have been identified as having a signific
 ant impact on admission chances, and enhancing them can greatly improve overall application competitiveness.
     -  Data Augmentation:- Holistic Profiles: Advocate for the collection of a broader range of data that goes beyond traditional academic 
metrics. This should include extracurricular achievements, personal statements, and diversity factors. By capturing 
a more comprehensive view of applicants' backgrounds and experiences, admissions committees can better assess the ov
 erall potential of each candidate.
 -  Additional Features:- Correlation Insights: Given the strong correlation among CGPA, it would be beneficial to enrich the predictive m
 odel with a variety of diverse features.
-  These may include:
     -   Research Experience: Highlighting involvement in research projects or publications can showcase an applican
 t's commitment and analytical skills.

## Streamlit App

Find the StreamLit App here :https://entryprediction.streamlit.app/
