--------------Travel Itineraries with Recommender and Retrieval Augmented Generation--------------------------
-------------------------Alexander Mentzelopoulos-------------------------------------------------------------
----------------------------University of Bath 2024-----------------------------------------------------------


Contained here are all the files, notebooks, and data for generating Athens travel itineraries.

There are two parts to the system.

-------------------Part 1 Vectorize and Cluster---------------------------------------------------------------

The first part is clustering and embedding the raw data which is done in the 'Vectorize and Cluster' notebook.
Using the already preprocessed data in the Preprocessed Data folder, you can repeat the process and generate 
clusters for Accommodations, Hotels and Attractions following this notebook. 
Be sure to import the 'Embed and Cluster.py file' as this contains all the necessary functions.


--------------------------------------------------------------------------------------------------------------
-------------------Part 2 Itinerary Generation with Gemini----------------------------------------------------

The second part is generating travel itineraries from the embedded and clustered data from Part 1. This data is
already provided to the user in the 'Clustered Data' folder. Using this data and the 'Embed and Cluster.py' file
you can follow along the in the 'Itinerary Generation with Gemini' notebook, reviewing the saved output.

*************************************************************************************************************
*****You will not be able to run the calls to the Gemini API unless you create your own private API key******
************************************ and run the file in Google CoLab****************************************
*****For this reason we have saved the file with the output already generated for your convenience.**********
*************************************************************************************************************


--------------------------------------------------------------------------------------------------------------
-------------------Other Files--------------------------------------------------------------------------------

We have also included our first attempts to cluster the data in the 'Clustering with Metadata' folder where you
can follow along our early data exploration and data engineering process.
Lastly as a reminder, the Embed and Cluster.py file contains all the necessary functions to run the notebooks.

If you would like a personalized travel itinerary yourself, please let me know and fill out the survey link
(in the appendix of the report)and you will recieve a personalized Athens travel itinerary.
