import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

#############################################################
#################### Load the Model #########################
#############################################################

filepath_ = r"adaBoost_model.sav"
adaBoost_model = pickle.load(open(filepath_, 'rb'))

#############################################################
#################### Prediction Function ####################
#############################################################

def diabetes_prediction(age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, gential_thrush, visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity):
    
    # change the following attributes from categorical into numerical form     
    if gender == "Male":
        gender = 1
    else:
        gender = 0
    
    if polyuria == "Yes":
        polyuria = 1
    else:
        polyuria = 0
    
    if polydipsia == "Yes":
        polydipsia = 1
    else:
        polydipsia = 0
    
    if sudden_weight_loss == "Yes":
        sudden_weight_loss = 1
    else:
        sudden_weight_loss = 0
    
    if weakness == "Yes":
        weakness = 1
    else:
        weakness = 0
    
    if polyphagia == "Yes":
        polyphagia = 1
    else:
        polyphagia = 0
        
    if gential_thrush == "Yes":
        gential_thrush = 1
    else:
        gential_thrush = 0
    
    if visual_blurring == "Yes":
        visual_blurring = 1
    else:
        visual_blurring = 0
    
    if itching == "Yes":
        itching = 1
    else:
        itching = 0
    
    if irritability == "Yes":
        irritability = 1
    else:
        irritability = 0
    
    if delayed_healing == "Yes":
        delayed_healing = 1
    else:
        delayed_healing = 0
    
    if partial_paresis == "Yes":
        partial_paresis = 1
    else:
        partial_paresis = 0
    
    if muscle_stiffness == "Yes":
        muscle_stiffness = 1
    else:
        muscle_stiffness = 0
    
    if alopecia == "Yes":
        alopecia = 1
    else:
        alopecia = 0
    
    if obesity == "Yes":
        obesity = 1
    else:
        obesity = 0
    
    # combine the transformed input into an array list
    input_data = np.array([age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, gential_thrush, visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity]).reshape(1, -1)

    # predict the class 
    predictClass = adaBoost_model.predict(input_data)
    # predict the probability for the class
    predictProbaClass = adaBoost_model.predict_proba(input_data)
    
    return predictProbaClass, predictClass



########################################################
#################### MAIN PAGE #########################
########################################################


def main():
    st.set_page_config(layout = "wide", 
                       page_title = "Diabetes Prediction")
    
    
    st.title("Are you at risk of developing diabetes?")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("General Information ")
        #st.image("images/information.jpg", width = 720)
        
        with st.expander("What is diabetes?"):
            #st.header("Overview of Diabetes")
            st.image("images/overview.jpg", width = 500)
            st.write("""
                     **Diabetes** is a chronic, metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which leads over time to serious damage to the heart, 
                     blood vessels, eyes, kidneys and nerves. The most common is type 2 diabetes, usually in adults, which occurs when the body becomes resistant to insulin or doesn't make enough insulin. 
                     In the past three decades the prevalence of type 2 diabetes has risen dramatically in countries of all income levels. Type 1 diabetes, once known as juvenile diabetes or insulin-dependent 
                     diabetes, is a chronic condition in which the pancreas produces little or no insulin by itself. For people living with diabetes, access to affordable treatment, including insulin, is critical 
                     to their survival. There is a globally agreed target to halt the rise in diabetes and obesity by 2025. 
                     """)
                     
        with st.expander("What is the motivation?"):
            #st.header("Motivation")
            st.image("images/motivation.png", width = 500)
            st.write("""
                     **Diabetes** is one of the fastest growing chronic life threatening diseases that have already affected 422 million people worldwide according
                     to the report of World Health Organization (WHO), in 2018. Due to the presence of a relatively long asymptomatic phase, early detection of diabetes is always desired for
                     a clinically meaningful outcome. Around 50% of all people suffering from diabetes are undiagnosed because of its long-term asymptomatic phase.
                     """)
        
        with st.expander("What is so unique about this health web app?"):
            #st.header("What is so unique about this web app?")
            st.image("images/unique.jpg", width = 500)
            st.write("""
                     This web app comes with an integration of an ensemble algorithm - AdaBoost Classifier to predict the classification of diabetes or non-diabetes. The Adaboost Classifier was
                     trained using 80% of the original dataset and the remaining 20% of the original dataset was used to evaluate the performance of the proposed model. Originally, the dataset consists of 520 observations with
                     17 characteristics, collected using direct questionnaires and diagnoses results from the patients in the Sylhet Diabetes Hospital in Sylhet, Bangladesh. 
                     """)
        
        with st.expander("What are the common symptoms? "):
            #st.header("What are the common symptoms?")
            st.image("images/symptoms.jpg", width = 500)
            st.write("""
                     Symptoms of type 1 diabetes include the need to urinate often, thirst, constant hunger, weight loss, vision changes and fatigue. 
                     These symptoms may occur suddenly. Symptoms for type 2 diabetes are generally similar to those of type 1 diabetes, but are often less marked. As a result, the disease may be diagnosed several years after onset, after complications have already arisen. For this reason, it is important to be aware of risk factors. 
                     """)
        
        with st.expander("How do we prevent diabetes?"):
            #st.header("How do we prevent diabetes?")
            st.image("images/prevention.jpg", width = 500)
            st.write("""
                     Type 1 diabetes cannot currently be prevented. Effective approaches are available to prevent type 2 diabetes and to prevent the complications and premature death that can result from all types of diabetes. 
                     These include policies and practices across whole populations and within specific settings (school, home, workplace) that contribute to good health for everyone, 
                     regardless of whether they have diabetes, such as exercising regularly, eating healthily, avoiding smoking, and controlling blood pressure and lipids. 
                     """)
            st.write("""
                     The starting point for living well with diabetes is an early diagnosis – the longer a person lives with undiagnosed and untreated diabetes, the worse their health outcomes are likely to be. 
                     Easy access to basic diagnostics, such as blood glucose testing, should therefore be available in primary health care settings. 
                     Patients will need periodic specialist assessment or treatment for complications. 
                     """)
            st.write("""
                     A series of cost-effective interventions can improve patient outcomes, regardless of what type of diabetes they may have. These interventions include blood glucose control, through 
                     a combination of diet, physical activity and, if necessary, medication; control of blood pressure and lipids to reduce cardiovascular risk and other complications; 
                     and regular screening for damage to the eyes, kidneys and feet, to facilitate early treatment. 
                     """)
        
        with st.expander("What is body mass index (BMI)? "):
            st.image("images/bmi.jpg")
            st.write("""
                     **Body mass index (BMI)** is a person's weight in kilograms divided by the square of height in metres. BMI is an inexpensive and easy screening method for weight category - underweight, healthweight, overweight, and obesity. 
                     """)
            st.write("""
                     BMI does not measure body fat directly, bmi BMI is moderately correlated with more direct measures of body fat. Furthermore, BMI appears to be more correlated with various metabolic and disease outcome. 
                     """)
            
            st.subheader("BMI Calculator")
            st.write("You may use the BMI calculator below to determine your BMI and weight category. ")
            st.info("NOTE: All fields are mandatory.")
            
            weight = st.number_input("Enter your weight in kilograms: ", min_value = 0.00, format = "%.2f")
            height = st.number_input("Enter your height in metres: ", min_value = 0.00, max_value = 3.00, format = "%.2f")
            
            # prompt the user to enter the button to calculate bmi
            bmi_button = st.button("Calculate BMI")
            
            if bmi_button:
                # compute BMI 
                if weight == 0 and height == 0: 
                    st.error("WARNING: Both weight and height must be above zero. ")
                elif weight == 0:
                    st.error("WARNING: Weight must be above zero. ")
                elif height == 0:
                    st.error("WARNING: Height must be above zero.")
                else: 
                    bmi = weight/(height * height)
                    
                    if bmi < 18.5:
                        st.warning("Your BMI is {:.2f} and you are classified as underweight.".format(bmi))
                    elif bmi >= 18.5 and bmi <= 24.9:
                        st.success("Your BMI is {:.2f} and you are classified as normal.".format(bmi))
                    elif bmi >= 25.0 and bmi <= 29.9:
                        st.warning("Your BMI is {:.2f} and you are classified as overweight. ".format(bmi))
                    elif bmi >= 30.0 and bmi <= 34.9:
                        st.error("Your BMI is {:.2f} and you are classified as obese.".format(bmi))
                    else:
                        st.error("Your BMI is {:.2f} and you are classified as extremely obese.".format(bmi))
            
        st.header("Reference Links")
        st.write("1. https://www.who.int/health-topics/diabetes#tab=tab_1")
        st.write("2. https://www.who.int/health-topics/diabetes#tab=tab_2")
        st.write("3. https://www.who.int/health-topics/diabetes#tab=tab_3")
        st.write("4. https://www.kaggle.com/andrewmvd/early-diabetes-classification")
        st.write("5. https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html")
        
        
        
    with col2:
        st.header("Early Diabetes Diagnosis ")
        st.image("images/diabetes.jpg", width = 760)
        st.warning("""**MEDICAL DISCLAIMER:** This health web app does not contain medical device and has not been approved by healthcare provider or other physician. 
                   Therefore, this health web app is meant for **informational and educational purposes** only. Furthermore, this health web app is not intended 
                   to substitute for professional medical advice, diagnosis, or treatment. Please consult your physician for personalized medical advice. Always seek 
                   the advice of a physician or other healthcare provider with any questions regarding a medical condition. 
                """)
                
        st.info("""
                **NOTE:** All fields are mandatory. 
                """)
        
        st.subheader("Personal Information")
        # prompt the user to provide personal information such as name 
        name = st.text_input("Enter your full name (as per in NRIC): ", placeholder = "E.g. James Phua", max_chars = 30)
        
        # prompt the user to select citizenship
        citizenship = st.radio("Select your citizenship: ", ['Singapore Citizen', 'Permanent Resident', 'Foreigner with Long-Term Pass'])
        
        # prompt the user to select age from the slider
        age = st.slider('Select your age (as of today): ', min_value = 1, max_value = 100)
        
        # prompt the user for gender
        gender = st.radio("Choose your gender: ", ['Male', 'Female'])
        
        # prompt the user to enter personal email address
        email = st.text_input("Enter your designated personal email address: ", placeholder = "E.g. jamesphua85@gmail.com", max_chars = 110)
        
        # prompt the user whether would like to proceed with diagnosis
        st.warning("""**IMPORTANT POLICY:** 
                   Under no circumstances will the organization be held responsible or liable in any way for misdiagnosis as this web app
                 is only intended for research and academic purposes. Please **tick** the checkbox if you agree with the policy and be willing to use this health web app for
                 diabetes screening at your own risk. """)
        proceed1 = st.checkbox("Yes, I agree with the policy and be willing to use this health web app for diabetes screening at my own risk. ")
        proceed2 = st.checkbox("No, I disagree with the policy and be unwilling to use this health web app for diabetes screening at my own risk. ")
        
        st.write("---")
        
        if proceed1 and proceed2:
            st.error("WARNING: Choose either option!")
            
        elif proceed1:
            
            st.subheader("Questionnaire")
            
            st.info("**NOTE:** Please kindly complete this questionnaire and then click on the 'Predict' button at the end of this questionnaire to predict your diabetes diagnosis. ")
            
            # prompt the user whether experienced excessive urination or not
            polyuria = st.radio("1. Do you experienced of excessive urination? ", ['Yes', 'No'])
            
            # prompt the user whether experienced excessive thirst or excessive drinking or not 
            polydipsia = st.radio("2. Do you experienced of feeling thirst or excessive drinking of water?", ['Yes', 'No'])
            
            # prompt the user whether had an episode of sudden weight loss or not
            sudden_weight_loss = st.radio("3. Do you had an episode of sudden weight loss? ", ['Yes', 'No'])
            
            # prompt the user whether had an episode of sudden weakness
            weakness = st.radio("4. Do you had an episode of feeling sudden weakness? ", ['Yes', 'No'])
            
            # prompt the user whether had an episode of excessive/extreme hunger 
            polyphagia = st.radio("5. Do you had an episode of feeling excessive/ extreme hunger? ", ['Yes', 'No'])
            
            # prompt the user had a yeast infection 
            gential_thrush = st.radio("6. Do you had a yeast infection in the past? ", ['Yes', 'No'])
            
            # prompt the user had an episode of blurred vision 
            visual_blurring = st.radio("7. Do you had an episode of blurred vision? ", ['Yes', 'No'])
            
            # prompt the user had an episoe of itch
            itching = st.radio("8. Do you had an episode of itch? ", ['Yes', 'No'])
            
            # prompt the user had an episode of irritability
            irritability = st.radio("9. Do you had an episode of irritability? ", ['Yes', 'No'])
            
            # prompt the user has a delayed healing
            delayed_healing = st.radio("10. Do you had an noticed delayed healing when wounded? ", ['Yes', 'No'])
            
            # prompt the user had an episode of weakening of muscles 
            partial_paresis = st.radio("11. Do you had an episode of weakening of muscles or group of muscles? ", ['Yes', 'No'])
            
            # prompt the user had an episode of muscles stiffness
            muscle_stiffness = st.radio("12. Do you had an episode of muscle stiffness? ", ['Yes', 'No'])
            
            # prompt the user had experienced hair loss
            alopecia = st.radio("13. Do you experienced hair loss? ", ['Yes', 'No'])
            
            # prompt the user whether obese or not 
            obesity = st.radio("14. Are you obese based on your body mass index?", ['Yes', 'No'])
            
            # prompt the user to click on the button to perform predict
            predict_button = st.button("Predict")
            
            
            
            if predict_button:
                if name == "" and id == "" and email == "":
                    st.error("WARNING: Your full name, identification number, and personal email address are empty.")
                elif name == "" and id == "":
                    st.error("WARNING: Your full name and identification number are empty.")
                elif name == "" and email == "":
                    st.error("WARNING: Your full name and personal email address are empty.")
                elif id == "" and email == "":
                    st.error("WARNING: Your identification number and personal email address are empty.")
                elif name == "":
                    st.error("WARNING: Your full name cannot be empty.")
                elif id == "":
                    st.error("WARNING: Your identification number cannot be empty.")
                elif email == "":
                    st.error("WARNING: Your personal email address cannot be empty.")
                elif name.replace(" ", "").isalpha() == False:
                    st.error("WARNING: Your full name must be in alphabets. ")
                else:
                    with st.spinner("In Progress..."):
                        time.sleep(5)
                    # call the function to perform prediction on diabetes 
                    predicted_proba_, predicted_class_ = diabetes_prediction(age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, gential_thrush, visual_blurring, itching, irritability, delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity)
                    
                    st.write("---")
                    st.subheader("Probabilities Outcomes for Diabetes and Non-Diabetes")
                    st.write("Hi {} ({})! Here are your predicted probabilities outcomes for both diabetes and non-diabetes. ".format(name, citizenship))
                    predicted_proba_df = pd.DataFrame(predicted_proba_*100, columns = ['Non-Diabetes (%)', 'Diabetes (%)'])
                    st.dataframe(predicted_proba_df)
                    
                    st.subheader("Diabetes Diagnosis Outcome")
                    
                    if predicted_class_ == 0:
                        st.success("Hi {}! You have about {:.2f}% chance of being diagnosed with non-diabetes.".format(name, float(predicted_proba_[:,0]*100)))
                    else:
                        st.error("Hi {}! You have about {:.2f}% chance of being diagnosed with diabetes.".format(name, float(predicted_proba_[:,1]*100)))
                        
                    st.subheader("General Remarks")
                    st.warning("""Please be informed that the above diagnosis outcome is just preliminary diagnosis and should not be used as a substitute for professional medical advice. 
                               As such, a qualified physician should make a decision based on each person's medical history and symptoms.  """)
                    st.error("**NOTE:** Please seek immediate medical attention at your nearest clinics or hospitals if your symptoms have not been improving or have been worsening.  ")
                    st.info("Thank you for using our health web app! Have a great day ahead! ")
                    
        elif proceed2:
            st.info("Thank you for visting our health web app! Have a nice day! ")
        
if __name__ == "__main__":
    main()