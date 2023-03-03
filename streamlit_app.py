import pandas as pd
import plotly.express as px
from datetime import date, datetime
import numpy as np
import streamlit as st


def get_rfm_scores(data):
    '''
    function to get the rfm score for each agency
    '''
    
    # log transform to deal with long tail
    data['rec_log']  = np.log(data['rec_week'])
    data['rel_freq_log']  = np.log(data['rel_freq'])
    data['mon_log'] = np.log(data['mon'])
    
    # compute individual scores
    data["rec_score"] = pd.qcut(data["rec"], 3, labels=[3, 2, 1]).astype("int")
    data["freq_score"] = pd.qcut(data["rel_freq_log"], 3, labels=[1, 2, 3]).astype("int")
    data["mon_score"] = pd.qcut(data["mon_log"], 3, labels=[1, 2, 3]).astype("int")
    
    # compute RFM score
    data['RFM'] = data["rec_score"] + data["freq_score"] + data["mon_score"]
    
    return data


def get_rfm_data_last_q(regenerate=False, start_date='2022-10-01', end_date='2022-12-31'):
    '''
    function to get final rfm data for the last quarter
    '''
    data = pd.read_csv("rfm_data_q3_2022_week.csv")
    return data


def get_l2b_data(regenerate=False, start_date='2022-10-01', end_date='2022-12-31'):
    '''
    function to get l2b data for the last quarter
    '''
    
    l2b_data = pd.read_csv("l2b_data_q3_2022.csv")
    return l2b_data


def get_rfm_data_curr_q(regenerate=False, start_date='2023-01-01', end_date='2023-02-15'):
    '''
    function to get final rfm data for the the current quarter
    '''
    data = pd.read_csv("rfm_data_curr.csv")
    return data


def get_rooms_pax_conversion_data(regenerate=False):
    
    rooms_pax_conversion_legacy = pd.read_csv("rooms_pax_conversion_legacy.csv")
    rooms_pax_conversion = pd.read_csv("rooms_pax_conversion.csv")
    
    return rooms_pax_conversion_legacy, rooms_pax_conversion



data = get_rfm_data_last_q(regenerate=False)   
data_curr = get_rfm_data_curr_q(regenerate=False)   
l2b_data = get_l2b_data(regenerate=False)
rooms_pax_conversion_legacy, rooms_pax_conversion = get_rooms_pax_conversion_data(regenerate=False)

rfm_cat_mapping = {9: 'A', 8: 'B', 7: 'C', 6: 'D', 5: 'E', 4: 'F', 3: 'G'}
data['RFM_cat'] = data['RFM'].apply(lambda x: rfm_cat_mapping[x])
data_curr['RFM_cat'] = data_curr['RFM'].apply(lambda x: rfm_cat_mapping[x])

regions_list = list(data['region'].unique())
rfm_list = sorted(list(data['RFM_cat'].unique()))


with st.sidebar:
    region = st.selectbox('Select a Region: ', regions_list)
    st.write("")
    content = st.radio('View:', ['RFM Cohorts', 'Cohort Movement', 'Complete Data'])
    rfm_cat_mapping_df = pd.DataFrame({'RFM': list(rfm_cat_mapping.keys()), 'RFM_cat': list(rfm_cat_mapping.values())})
    st.write("")
    st.write("RFM Mapping: ")
    st.dataframe(rfm_cat_mapping_df)


data_region = data.query(f"region=='{region}'").sort_values(['RFM_cat'])
data_curr_region = data_curr.query(f"region=='{region}'").sort_values(['RFM_cat'])

rfm_data_l2b = pd.DataFrame(columns=['RFM_cat', 'L2B %', 'searches', 'bookings'])
for rfm in rfm_list:
    agencies_list = list(data_region.query(f"RFM_cat == '{rfm}'")['agency_id'].unique())
    if len(agencies_list)>0:
        num_searches = sum(l2b_data.query(f"agency_id in @agencies_list")['num_searches'])
        num_bookings = sum(l2b_data.query(f"agency_id in @agencies_list")['num_bookings']) 
        row = {'RFM_cat': rfm, 'L2B %': num_bookings/num_searches*100, 'searches': num_searches, 'bookings': num_bookings}
    else:
        continue
    rfm_data_l2b = rfm_data_l2b.append([row], ignore_index=True)
    
rfm_data_conv = pd.DataFrame(columns=['RFM_cat', 'conversion %', 'rooms', 'pax', ])
for rfm in rfm_list:
    agencies_list = list(data_region.query(f"RFM_cat == '{rfm}'")['agency_id'].unique())
    if len(agencies_list)>0:
        rooms_page_legacy = sum(rooms_pax_conversion_legacy.query(f"agency_id in @agencies_list")['rooms_page'])
        pax_page_legacy = sum(rooms_pax_conversion_legacy.query(f"agency_id in @agencies_list")['pax_page'])
        rooms_page = sum(rooms_pax_conversion.query(f"agency_id in @agencies_list")['rooms_page'])
        pax_page = sum(rooms_pax_conversion.query(f"agency_id in @agencies_list")['pax_page']) 
        row = {'RFM_cat': rfm, 'conversion %': (pax_page_legacy+pax_page)/(rooms_page_legacy+rooms_page)*100, 'rooms': rooms_page_legacy+rooms_page, 'pax': pax_page_legacy+pax_page}
    else:
        continue
    rfm_data_conv = rfm_data_conv.append([row], ignore_index=True)
        

if content == 'RFM Cohorts':    
    st.header(f"{region}: {content}")
    st.write("")
    st.write("Cohorts based on agency activity in the previous quarter: **2022/10/01 - 2022/12/31** ")
    st.write(f"Total number of Agencies: **{data_region['agency_id'].nunique()}**")
    region_rfm_hist = px.histogram(data_region, 'RFM_cat')
    # region_rfm_hist.update_layout(height=400, width=600)
    st.plotly_chart(region_rfm_hist)
    cols = st.columns([1.5, 1.5, 1])
    with cols[0]:
        cohort_details = data_region.groupby(['RFM_cat'], as_index=False).mean()[['RFM_cat', 'freq', 'mon', 'rec', 'days_active']]
        st.write("Mean RFM Values")
        st.dataframe(cohort_details.set_index(['RFM_cat'], drop=True).astype('int'))
    with cols[1]:
        st.write("L2B")
        # [['L2B %']]
        st.dataframe(rfm_data_l2b.set_index(['RFM_cat'], drop=True).astype('int'))
    with cols[2]:
        st.write("Rooms-Pax Conversion")
        # 
        st.dataframe(rfm_data_conv.set_index(['RFM_cat'], drop=True).astype('int')[['conversion %']])
    
    st.write("")
    st.write("")
    rfm = st.selectbox('Select an RFM Cohort to view Agencies: ', rfm_list)
    st.write("")
    st.write("")
    data_region_rfm = data_region.query(f"RFM_cat == '{rfm}'")
    st.write(f"Number of Agencies: **{data_region_rfm['agency_id'].nunique()}**")
    data_cols = ['agency_id', 'agency_name', 'created_on', 'first_overall_sale', 'first_q_sale', 'last_q_sale', 'freq', 'mon', 'rec', 'rec_week', 'days_active', 'weeks_active', 'rel_freq', 'new_agency', 'freq_score', 'mon_score', 'rec_score', 'RFM_cat']
    st.dataframe(data_region_rfm[data_cols].reset_index(drop=True))
    
    
if content == 'Cohort Movement':
    st.header(f"{region}: {content}")
    st.write("")
    st.write("New Cohorts based on agency activity in the first half of the current quarter: **2023/01/01 - 2023/02/15** ")
    st.write("")
    rfm_cat = st.selectbox('Select an RFM Cohort to view its movement: ', rfm_list)
    agencies_list = list(data_region.query(f"RFM_cat == '{rfm_cat}'")['agency_id'])
    data_curr_rfm = data_curr_region.query("agency_id in @agencies_list")
    active_agencies_list = list(data_curr_rfm['agency_id'].unique())
    inactive_agencies = data_region.query(f"RFM_cat == '{rfm_cat}' & agency_id not in @active_agencies_list")[['agency_id', 'agency_name', 'created_on', 'first_overall_sale']]
    st.write("")
    st.write("")
    st.write(f"Number of Agencies in this Cohort in the last quarter: **{len(agencies_list)}**")
    st.write(f"Number of Agencies from that Cohort that returned this quarter: **{data_curr_rfm['agency_id'].nunique()}**")
    curr_rfm_hist = px.histogram(data_curr_rfm, x='RFM_cat')
    st.plotly_chart(curr_rfm_hist)
    data_cols = ['agency_id', 'agency_name', 'created_on', 'first_overall_sale', 'first_q_sale', 'last_q_sale', 'freq', 'mon', 'rec', 'rec_week', 'days_active', 'weeks_active', 'rel_freq', 'new_agency', 'freq_score', 'mon_score', 'rec_score', 'RFM_cat']
    st.write("Agency details for the current quarter: ")
    st.dataframe(data_curr_rfm[data_cols].reset_index(drop=True))
    st.write("")
    st.write("Agencies that did not return in the current quarter: ")
    st.dataframe(inactive_agencies.reset_index(drop=True))
    

if content == 'Complete Data':
    st.header(f"{region}: {content}")
    data_cols = ['agency_id', 'agency_name', 'created_on', 'first_overall_sale', 'first_q_sale', 'last_q_sale', 'freq', 'mon', 'rec', 'rec_weeks', 'days_active', 'weeks_active', 'rel_freq', 'new_agency', 'freq_score', 'mon_score', 'rec_score', 'RFM_cat']
    st.write("")
    st.write("Last quarter cohort data: ")
    st.dataframe(data_region.drop(['rec_log', 'rel_freq_log', 'mon_log'], axis=1).reset_index(drop=True))
    st.write("")
    st.write("Current quarter cohort data: ")
    st.dataframe(data_curr_region.drop(['rec_log', 'rel_freq_log', 'mon_log'], axis=1).reset_index(drop=True))
    

    
