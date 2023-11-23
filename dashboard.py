import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from streamlit_option_menu import option_menu
import numpy as np

def customer_total(df):
    total_customer = df['customer_id'].nunique()
    return total_customer

def customer_unique_total(df):
    total_customer_unique = df['customer_unique_id'].nunique()
    return total_customer_unique

def customer_state(df):
    total_customer_state = df['customer_state_full'].nunique()
    return total_customer_state

def customer_city(df):
    total_customer_city = df['customer_city'].nunique()
    return total_customer_city

def customer_by_city(df):
    # To group customers customer_city, and customer_state
    result = df.groupby(['customer_city', 'customer_state_full'])['customer_id'].count().reset_index().sort_values(by='customer_id', ascending=False)
    result = result.rename(columns={'customer_id': 'amount_of_customers'})
    return result

def customer_by_state(df):
    total_customer_per_state = df.groupby('customer_state_full')['customer_id'].count().reset_index(name='amount_of_customers').sort_values(by='amount_of_customers', ascending=False)
    return total_customer_per_state

# seller
def seller_total(df):
    total_seller = df['seller_id'].nunique()
    return total_seller

def seller_state(df):
    total_seller_state = df['seller_state_full'].nunique()
    return total_seller_state

def seller_city(df):
    total_customer_city = df['seller_city'].nunique()
    return total_customer_city

def seller_by_city(df):
    # clculate the total seller in each city
    sellers_city_df = df.groupby(['seller_city', 'seller_state_full'])['seller_id'].count().reset_index(name='amount_of_sellers').sort_values(by='amount_of_sellers', ascending=False)
    return sellers_city_df

def seller_by_state(df):
    total_seller_per_state = df.groupby('seller_state_full')['seller_id'].count().reset_index(name='amount_of_sellers').sort_values(by='amount_of_sellers', ascending=False)
    return total_seller_per_state

def create_avg_revenue_per_seller(df):
    revenue = df[df['order_status']!='canceled']['price'].sum()
    total = seller_total(df[df['order_status']!='canceled'])
    revenue_per_seller = revenue/total
    return revenue_per_seller

# orders
def total_orders(df):
    total_order = df['order_id'].nunique()
    return total_order

def total_cancels(df):
    total_cancel = df[df['order_status'] == 'canceled']['order_id'].nunique()
    return total_cancel

def total_product_categories(df):
    total_product_category = df['product_category_name'].nunique()
    return total_product_category

def aov(df):
    aov_all = df['price'].sum()/df['order_id'].nunique()
    return aov_all

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    
    return daily_orders_df

def create_order_per_year(df):
    order_per_year = df.groupby(['order_purchase_year','customer_city', 'customer_state_full'])['order_id'].nunique().reset_index(name='count_order').sort_values(by='count_order', ascending=False)
    return order_per_year

def create_city_trend_order_df(df):
    # top 5 categories by city
    top_city = df['customer_city'].value_counts().nlargest(5).index
    # Category per year is only for the top 5 categories.
    order_per_year = create_order_per_year(df)
    top_city_trend = order_per_year[order_per_year['customer_city'].isin(top_city)]
    return top_city_trend

def create_state_trend_order_df(df):
    # top 5 categories by city
    top_state = df['customer_state_full'].value_counts().nlargest(5).index
    # Category per year is only for the top 5 categories.'
    order_per_year = create_order_per_year(df).groupby(['customer_state_full','order_purchase_year'])['count_order'].sum().reset_index()
    top_state_trend = order_per_year[order_per_year['customer_state_full'].isin(top_state)]
    return top_state_trend

def create_product_category_df(df):
    # calculate amount of item by product category
    product_category_df = df.groupby(['product_category_name'])['order_id'].count().reset_index(name='amount_of_item').sort_values(by='amount_of_item', ascending=False)
    return product_category_df

def create_category_trend(df):
    # top 5 categories by product category name
    top_categories = df['product_category_name'].value_counts().nlargest(5).index
    product_sales_per_year = df.groupby(['order_purchase_year','product_category_name'])['order_id'].nunique().reset_index(name='total_sales').sort_values(by='total_sales', ascending=False)
    # Filter DataFrame only top 5 categories
    top_category_trend = product_sales_per_year[product_sales_per_year['product_category_name'].isin(top_categories)]
    return top_category_trend

def create_canceled_by_product_category(df):
    canceled_by_product_category = df.groupby(df[df['order_status'] == 'canceled']['product_category_name'])['order_id'].count().reset_index(name='amount_of_item_canceled').sort_values(by='amount_of_item_canceled', ascending=False)
    return canceled_by_product_category

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max", # get the date last order
        "order_id": "nunique", # calculate the total unique order
        "price": "sum" # calculate revenue
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

    # calculate the last order per customers
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = max_date.date() # last date order
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days) # calculate last date order - max order timestamp

    rfm_df.drop("max_order_timestamp", axis=1, inplace=True) # drop column max_order_timestamp
    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)  # The smaller the recency, the better
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)  # The larger the frequency, the better
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)  # The larger the monetary, the better

        # normalizing the rank of the customers
    rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*100
    rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*100
    rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*100

    rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)
    # Calculate RFM score
    # For recency, the weight is 0.15
    # For frequency, the weight is 0.28
    # For monetary, the weight is 0.57
    rfm_df['RFM_score'] = 0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']
    rfm_df['RFM_score'] *= 0.05  # The maximum RFM score is 5
    rfm_df = rfm_df.round(2)  # Rounding to 2 decimal places
    
    # Customer segment categories from RFM score
    # Define the segmentation based on RFM scores
    rfm_df['customer_segment'] = np.where(
        rfm_df['RFM_score'] > 4.5, "Top customers", (np.where(
            rfm_df['RFM_score'] > 4, "High value customer",(np.where(
                rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
                    rfm_df['RFM_score'] > 1.6, 'Low value customers', 'Lost customers'))))))
    
    # Menentukan urutan kategorikal
    segment_order = ["Lost customers", "Low value customers", "Medium value customer", "High value customer", "Top customers"]

    # Mengubah kolom customer_segment menjadi tipe data kategorikal dengan urutan yang diinginkan
    rfm_df['customer_segment'] = pd.Categorical(rfm_df['customer_segment'], categories=segment_order, ordered=True)
    return rfm_df

def create_rfm_categorical(rfm_df):
    # calculate the total customer based on customer segment
    customer_segment_df = rfm_df.groupby(by="customer_segment", as_index=False).customer_id.nunique()
    return customer_segment_df

################################# Customers ###########################################
def customer_analysis():
    st.subheader("Customers Analysis")

    # menyiapkan metrics customer
    total_customer = customer_total(all_df)
    total_customer_unique = customer_unique_total(all_df)
    total_customer_state = customer_state(all_df)
    total_customer_city = customer_city(all_df)

    total_customer_per_city = customer_by_city(all_df)
    total_customer_per_state = customer_by_state(all_df)

    # Membagi layout menjadi beberapa column
    col2, col3, col4, col5 = st.columns(4)  # Gunakan beta_columns untuk membuat 2 kolom
    # kolom 1 baris 1
    with col2:
            st.info('Total Customers',icon="🦲")
            st.metric("all customer", value=total_customer)
    with col3:
            st.info('Total Customers Unique',icon="🦲")
            st.metric("nunique", value=total_customer_unique)
    with col4:
            st.info('Total State',icon="🏙️")
            st.metric("BRZ", value=total_customer_state)
    with col5:
            st.info('Total City',icon="🏙️")
            st.metric("BRZ", value=total_customer_city)


    # question 1
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(45, 8))
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    # question 1 : Customer by city
    sns.barplot(
        y="amount_of_customers",
        x="customer_city",
        data= total_customer_per_city.head(),
        palette=colors,
        ax=ax[0]
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Number of Customer by City", loc="center", fontsize=30)
    ax[0].tick_params(axis='y', labelsize=35)
    ax[0].tick_params(axis='x', labelsize=30)

    # question 1 : Customer by state
    sns.barplot(
        y="amount_of_customers",
        x="customer_state_full",
        data= total_customer_per_state.head(),
        palette=colors,
        ax=ax[1]
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("Number of Customer by State", loc="center", fontsize=30)
    ax[1].tick_params(axis='y', labelsize=35)
    ax[1].tick_params(axis='x', labelsize=30)

    st.pyplot(fig)  

    # Question 7 - 9 : RFM
    # grouping by customer'
    st.subheader("RFM Analysis")
    rfm_df = create_rfm_df(all_df)

    col2, col3, col4 = st.columns(3)
    with col2:
            st.info('Average Recency (days)',icon="📅")
            avg_recency = round(rfm_df.recency.mean(), 1)
            st.metric("avg", value=avg_recency)
    with col3:
            st.info('Average Frequency',icon="📊")
            avg_frequency = round(rfm_df.frequency.mean(), 2)
            st.metric("Avg", value=avg_frequency)
    with col4:
            st.info('Average Monetary',icon="💰")
            avg_monetary = format_currency(rfm_df.monetary.mean(), "R$", locale='pt_BR') 
            st.metric("avg", value=avg_monetary)

    customer_segment_df = create_rfm_categorical(rfm_df)

    fig, ax = plt.subplots(figsize=(25, 8))
    colors_ = ["#D3D3D3", "#72BCD4", "#72BCD4", "#D3D3D3", "#D3D3D3"]

    sns.barplot(
        x="customer_id",
        y="customer_segment",
        data=customer_segment_df.sort_values(by="customer_segment", ascending=False),
        palette=colors_
    )
    ax.set_title("Number of Customer for Each Segment", loc="center", fontsize=20)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='y', labelsize=20)
    st.pyplot(fig) 
    
    col1, col2 = st.columns((1, 2))
    with col1:
        selected_category = st.selectbox("Sort by:", ["frequency", "recency", "monetary",'customer_segment'])
        # Membuat radio button untuk memilih ascending atau descending
        sort_order = st.radio("Sort Order:", ["Ascending", "Descending"])
               
        # Mengurutkan DataFrame berdasarkan kategori yang dipilih dan arah pengurutan
        ascending_order = (sort_order == "Ascending")
        sorted_rfm_df = rfm_df.sort_values(by=selected_category, ascending=ascending_order)
        st.markdown(
            """
            <div style="background-color: #f4f4f4; padding: 10px; border-radius: 10px;">
                <p>RFM > 4.5 adalah Top Customers</p>
                <p>RFM > 4 adalah High Value Customer</p>
                <p>RFM > 3 Medium Value Customers</p>
                <p>RFM > 1.6 Low Value Customers</p>
                <p>RFM <= 1.6 Lost Customers</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        with col2:
            st.write("Result : ")
            st.dataframe(sorted_rfm_df[['customer_id','frequency', 'monetary', 'recency', 'RFM_score','customer_segment']], use_container_width=True)
            
#################################################### SELLER #################################
def seller_analysis():
    st.subheader("Sellers Analysis")

    # menyiapkan seller
    total_seller = seller_total(all_df)
    total_seller_state = seller_state(all_df)
    total_seller_city = seller_city(all_df)

    total_seller_per_city = seller_by_city(all_df)
    total_seller_per_state = seller_by_state(all_df)
    avg_revenue_per_seller = create_avg_revenue_per_seller(all_df)

    # Membagi layout menjadi beberapa column
    col1, col2, col3, col4 = st.columns(4)  

    with col1:
            st.info('Total Sellers',icon="🦲")
            st.metric("cnt", value=total_seller)
    with col2:
            st.info('Total State',icon="🏙️")
            st.metric("nunique", value=total_seller_state)
    with col3:
            st.info('Total City',icon="🏙️")
            st.metric("nunique", value=total_seller_city)
    with col4:
        st.info('Average Revenue Per Seller',icon="💰")
        st.metric("avg", value=format_currency(avg_revenue_per_seller, "R$", locale='pt_BR'))

    # Question 2 Seller
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(45, 8))
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    # question 2 : seller by city
    sns.barplot(
        y="amount_of_sellers",
        x="seller_city",
        data= total_seller_per_city.head(),
        palette=colors,
        ax=ax[0]
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Number of Seller by City", loc="center", fontsize=30)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].tick_params(axis='x', labelsize=25)

    # question 2: Seller by state
    sns.barplot(
        y="amount_of_sellers",
        x="seller_state_full",
        data= total_seller_per_state.head(),
        palette=colors,
        ax=ax[1]
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("Number of Seller by State", loc="center", fontsize=30)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].tick_params(axis='x', labelsize=25)

    st.pyplot(fig)

############################################## Orders ####################################
def order_analysis():
    # Orders
    st.subheader("Orders")

    total_order = total_orders(all_df)
    total_cancel = total_cancels(all_df)
    total_product_category = total_product_categories(all_df)
    aov_total = aov(all_df)

    daily_orders_df = create_daily_orders_df(all_df)

    # Membagi layout menjadi beberapa column
    col1, col2, col3, col4,col5 = st.columns(5)  

    with col2:
        st.info('Total Orders',icon="🛒")
        st.metric("sum", value=total_order)
    with col3:
        st.info('Total Canceled',icon="❌")
        st.metric("Total Canceled", value=total_cancel)
    with col4:
        st.info('Total Product Category',icon="🚀")
        st.metric("Total Product Category", value=total_product_category)
    with col5:
        st.info('Average Order Value',icon="💰")
        aov_total = format_currency(aov_total.round(2), "R$", locale='pt_BR') 
        st.metric("AOV", value=aov_total)
    with col1:
        st.info('Total Revenue',icon="💰")
        total_revenue = format_currency(daily_orders_df['revenue'].sum().round(2), "R$", locale='pt_BR') 
        st.metric("sum", value=total_revenue)

    # Question 3 : Order setiap tahun

    fig, ax = plt.subplots(figsize=(32, 8))
    ax.plot(
        daily_orders_df["order_purchase_timestamp"],
        daily_orders_df["order_count"],
        marker='o', 
        linewidth=2,
        color="#90CAF9"
    )
    ax.set_title("Number of Daily Order", loc="center", fontsize=30, pad=20)
    ax.tick_params(axis='y', labelsize=25)
    ax.tick_params(axis='x', labelsize=15)

    st.pyplot(fig)

    # Question 7 : Waktu orders

    time_category_counts = all_df['purchase_time_category'].value_counts()
    # Hitung jumlah pembelian untuk setiap hari
    day_counts = all_df['purchase_day'].value_counts()

    # Plot pie charts dalam subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3","#D3D3D3", "#D3D3D3"]
    colors2 = ["#72BCD4", "#D3D3D3", "#D3D3D3"]
    # Pie chart untuk distribusi pembelian berdasarkan hari
    ax[0].pie(day_counts, labels=day_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax[0].set_title('Distribution of Purchases by Day')

    # Pie chart untuk distribusi pembelian berdasarkan kategori waktu
    ax[1].pie(time_category_counts, labels=time_category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors2)
    ax[1].set_title('Distribution of Purchases by Time of Day')
    st.pyplot(fig)

    # Question 3 : Order by city
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(45, 8))

    sns.lineplot(
        y="count_order",
        x="order_purchase_year",
        data= create_city_trend_order_df(all_df),
        hue='customer_city',
        marker='o',
        ax=ax[0]
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Trend of Number of Order per Category Yearly (Top 5 Categories) by City", loc="center", fontsize=30, pad=20)
    ax[0].tick_params(axis='y', labelsize=25)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].legend(fontsize='25')

    # question 3 : Order by state
    sns.lineplot(
        y="count_order",
        x="order_purchase_year",
        data= create_state_trend_order_df(all_df),
        hue='customer_state_full',
        marker='o',
        ax=ax[1]
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].set_title("Trend of Number of Order per Category Yearly (Top 5 Categories) by State", loc="center", fontsize=30,pad=20)
    ax[1].tick_params(axis='y', labelsize=25)
    ax[1].tick_params(axis='x', labelsize=25)
    ax[1].legend(fontsize='25')

    st.pyplot(fig)
    st.subheader('Product')
    # Question 5 :

    # Plotting the trend of product categories per year for the top categories.
    fig, ax = plt.subplots(figsize=(32, 8))

    sns.lineplot(x='order_purchase_year', y='total_sales', hue='product_category_name', data=create_category_trend(all_df), marker='o')
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_title("Best Performing by Category Product", loc="center", fontsize=25, pad=20)
    ax.tick_params(axis ='y', labelsize=20)
    ax.tick_params(axis ='x', labelsize=20)
    ax.legend(fontsize='20')

    st.pyplot(fig)

    # Question 4 : 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))
    colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(y="product_category_name", x="amount_of_item", data=create_product_category_df(all_df).head(5), palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Best Performing by Category Product", loc="center", fontsize=20)
    ax[0].tick_params(axis ='y', labelsize=20)
    ax[0].tick_params(axis ='x', labelsize=20)

    sns.barplot(y="product_category_name", x="amount_of_item", data=create_product_category_df(all_df).sort_values(by='amount_of_item', ascending=True).head(), palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing by Category Product", loc="center", fontsize=20)
    ax[1].tick_params(axis ='y', labelsize=20)
    ax[1].tick_params(axis ='x', labelsize=20)

    plt.suptitle("Best and Worst Performing Category Product by Number of Item Sold", fontsize=25,y=1.02)

    st.pyplot(fig)

    # Question 6 : 
    fig, ax = plt.subplots(figsize=(32, 8))
    colors = ["red", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    
    sns.barplot(
        x="amount_of_item_canceled",
        y="product_category_name",
        data= create_canceled_by_product_category(all_df).head(5),
        palette=colors
    )
    ax.set_title("Number of Item Canceled by Product Category", loc="center", fontsize=25, pad=20)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis ='y', labelsize=20)
    ax.tick_params(axis ='x', labelsize=20)

    st.pyplot(fig)

#################################### START ###############################3
# Page setting
st.set_page_config(layout="wide", page_title="ecommerce analysis")
# Center the title
st.markdown('<h1 style="text-align: center;">Ecommerce Analysis</h1>', unsafe_allow_html=True)

######################### Load cleaned data #############################
order_items_csv = pd.read_csv("order_items.csv")
orders_csv = pd.read_csv("orders.csv")
all_df = pd.merge(order_items_csv, orders_csv, how='left', on='order_id')

datetime_columns = ["order_purchase_timestamp", "order_estimated_delivery_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Filter data
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

############################################### SIDEBAR ###########################
st.sidebar.image("logo.png",caption="")

filter_product = all_df['product_category_name'].unique()
filter_state =  all_df['customer_state_full'].unique()
with st.sidebar :
    start_date, end_date = st.date_input(
            label='Rentang Waktu',min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
########################################### RENTANG WAKTU ##############################
# Mendefinisikan tanggal start dan end
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Menambahkan satu hari pada end_date dan mengurangkan satu hari pada start_date
start_date = start_date - pd.DateOffset(days=1)
end_date = end_date + pd.DateOffset(days=1)

all_df = all_df[(all_df["order_purchase_timestamp"] >= (start_date)) & 
                (all_df["order_purchase_timestamp"] <= (end_date))]

#menu bar
def sideBar():
 with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Customers","Sellers","Orders"],
        icons=["people","house","cart"],
        menu_icon="cast",
        default_index=0
    )
 if selected=="Customers":
    customer_analysis()
 if selected=="Sellers":
    seller_analysis()
 if selected=="Orders":
    order_analysis()
sideBar()

st.sidebar.caption('Copyright © ciciyuriza 2023')