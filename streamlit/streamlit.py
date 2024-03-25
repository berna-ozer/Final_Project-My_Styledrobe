import streamlit as st
import pandas as pd
import random
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

if 'backup' not in st.session_state:
    st.session_state.backup = None

# wardrobe dataframe loading
if 'wardrobe_df' not in st.session_state:
    st.session_state.wardrobe_df = pd.read_csv('wardrobe.csv').sort_values(by='type')

types = ['Shirts', 'Blouses', 'Jumpsuits', 'Dresses', 'Tshirts', 'Sweaters',
       'Tops', 'Cardigans', 'Jackets', 'Sweatshirts', 'Gilets',
       'Trench coats', 'Quilted coats/Padded', 'Blazers', 'Suit jackets',
       'Coats', 'Trousers', 'Skirts', 'Jeans', 'Shorts', 'Hoodie', 'Bags',
       'Jewellery', 'Shoes', 'Belts', 'Sunglasses',
       'Foulards and scarves', 'Dresses and jumpsuits',
       'Wallets and cases', 'Sweaters and cardigans', 'Hats and caps',
       'Blouses and shirts', 'Joggers']

colors = ['Ecru', 'Pink', 'Blue', 'White', 'Black', 'Green', 'Grey', 'Navy',
       'Beige', 'Russet', 'Khaki', 'Purple', 'Brown', 'Red', 'Silver',
       'Sand', 'Orange', 'Turquoise', 'Burgundy', 'Gold', 'Yellow',
       'Charcoal', 'Leather', 'Nude']


# define model building function
def build_model():
    # Building Model
    # Base model for feature extraction
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Inputs
    input_image1 = layers.Input(shape=(224, 224, 3))
    input_image2 = layers.Input(shape=(224, 224, 3))
    input_category1 = layers.Input(shape=(len(types),))
    input_category2 = layers.Input(shape=(len(types),))
    input_color1 = layers.Input(shape=(len(colors),))
    input_color2 = layers.Input(shape=(len(colors),))

    # Image feature extraction
    image_feature_extractor = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
    ])

    image1_features = image_feature_extractor(input_image1)
    image2_features = image_feature_extractor(input_image2)
    
    # Combine all features
    combined_features = layers.concatenate([image1_features, image2_features, input_category1, input_category2, input_color1, input_color2])
    
    # Fully connected layers
    x = layers.Dense(512, activation='relu')(combined_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Model
    model = models.Model(inputs=[input_image1, input_image2, input_category1, input_category2, input_color1, input_color2], outputs=output)
    
    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# define image processing function
def preprocess_image(image_path):
    image = tf.io.read_file(image_path + '.jpg')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # Resizing the image to 224x224
    image = image / 255.0  # Normalizing to [0,1]
    return image

# define future extraction function
def process_row_for_prediction(row):
    # Process images
    image = preprocess_image(row['image_link'])
    combi_image = preprocess_image(row['combi_image_link'])
    
    # Recalculate unique values or pass them as parameters if this is computationally expensive
    type = tf.one_hot(row['type_indices'], depth=len(types))  
    combi_type = tf.one_hot(row['combi_type_indices'], depth=len(types))
    color = tf.one_hot(row['color_indices'], depth=len(colors))
    combi_color = tf.one_hot(row['combi_color_indices'], depth=len(colors))
    
    # Combine all features
    features = (image, combi_image, type, combi_type, color, combi_color)
    #labels = row['match']
    
    return features,None

def tf_dataset_for_prediction(df, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    ds = ds.map(process_row_for_prediction, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# define make combins function
def make_combins(selected_row, other_rows):
    combin = pd.DataFrame()
    for idx, row in other_rows.iterrows():  # Iterate over rows of the Serie 
        if row['type'] == selected_row['type']:
            continue
        bottom = ['Trousers','Skirts','Jeans','Joggers','Shorts','Jumpsuits','Dresses and jumpsuits','Dresses']
        top = ['Tops','Shirts','Sweaters','Dresses','Tshirts','Hoodie','Cardigans','Sweaters and cardigans','Blouses and shirts','Blouses','Sweatshirts','Dresses and jumpsuits','Jumpsuits']
        jackets = ['Jackets', 'Gilets','Trench coats', 'Quilted coats/Padded', 'Blazers', 'Suit jackets',
       'Coats']
        if row['type'] in bottom and selected_row['type'] in bottom:
            continue
        if row['type'] in top and selected_row['type'] in top:
            continue
        if row['type'] in jackets and selected_row['type'] in jackets:
            continue
        combination = {
            'image_link': selected_row['image_link'],
            'type': selected_row['type'],
            'color': selected_row['color'],
            'combi_image_link': row['image_link'],
            'combi_type':  row['type'],
            'combi_color': row['color']
        }  
        combin = pd.concat([combin, pd.DataFrame([combination])], ignore_index=True)
    return combin

# define model loading function
@st.cache_resource
def load_model():
    loaded_model = build_model()

    # Load model weights
    loaded_model.load_weights("my_model.weights.h5")

    return loaded_model

# Load the model
loaded_model = load_model()

# define find pairs function
def find_pairs(selected_row, other_rows):
    combins = make_combins(selected_row, other_rows)
    combins['type_indices'] = pd.factorize(combins['type'])[0]
    combins['color_indices'] = pd.factorize(combins['color'])[0]
    combins['combi_type_indices'] = pd.factorize(combins['combi_type'])[0]
    combins['combi_color_indices'] = pd.factorize(combins['combi_color'])[0]
    predict_ds = tf_dataset_for_prediction(combins)
    predictions = loaded_model.predict(predict_ds)

    predicted_matches = (predictions > 0.90).astype(int)
    combins['predicted_match'] = predictions

    matches = combins[predicted_matches.flatten() == 1]
    return matches

#define random combin making function                                        
def make_combins_day():
    # Select a random item from the wardrobe
    random_index = random.randint(0, st.session_state.wardrobe_df.shape[0] - 1)
    random_item = st.session_state.wardrobe_df.iloc[random_index]

    # Find pairs for the random item
    combins = find_pairs(random_item, st.session_state.wardrobe_df) 
    
    # Sort by 'predicted_match' column in descending order
    combins_sorted = combins.sort_values(by='predicted_match', ascending=False)
    
    # Drop duplicates based on 'combi_type', keeping the first occurrence (which will be the highest 'predicted_match' value)
    unique_combins = combins_sorted.drop_duplicates(subset=['combi_type'], keep='first')
    
    # Define the lists of categories
    categories = {
        'bottom': ['Trousers', 'Skirts', 'Jeans', 'Joggers', 'Shorts'],
        'top': ['Tops', 'Shirts', 'Sweaters', 'Tshirts', 'Hoodie', 'Cardigans', 'Sweaters and cardigans', 'Blouses and shirts', 'Blouses', 'Sweatshirts'],
        'jackets': ['Jackets', 'Gilets', 'Trench coats', 'Quilted coats/Padded', 'Blazers', 'Suit jackets', 'Coats'],
    }

    selected_items = []
    
    # Iterate through each category
    for category, items in categories.items():
        # Check if any item from the category exists in unique_combins
        selected_item = unique_combins[unique_combins['combi_type'].isin(items)]
        
        # If an item exists, keep the one with the highest 'predicted_match'
        if not selected_item.empty:
            selected_item = selected_item.sample(n=1, random_state=42).iloc[0]  # Set a random state for reproducibility
            selected_items.append(selected_item)
     
    # Create a list of all items in the specified categories
    category_items = [item for sublist in categories.values() for item in sublist]

    # Filter unique_combins to include items not in the specified categories
    other_items = unique_combins[~unique_combins['combi_type'].isin(category_items)]

    # Filter out dresses from other items
    dresses = ['Dresses', 'Dresses and jumpsuits', 'Jumpsuits']
    other_items = other_items[~other_items['combi_type'].isin(dresses)]
    
    random_item.rename({'type': 'combi_type', 'color': 'combi_color', 'image_link': 'combi_image_link'}, inplace=True)

    # Concatenate the selected items with other_items and random_item
    result = pd.concat([pd.DataFrame([random_item]), pd.DataFrame(selected_items),other_items], ignore_index=True)
    
    if result.shape[0]<3:
        return make_combins_day()
    else:
        return result

# define spacing function
def spacing():
    st.write("")  # Add an empty line for spacing

def style_button():
    st.markdown(
        """
        <style>
        button {
            padding-top: 1px !important;
            padding-bottom: 1px !important;
            padding-left: 6px !important;
            padding-right: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Initialize save_list if it doesn't exist
if 'save_list' not in st.session_state:
    st.session_state.save_list = pd.read_csv('my library/save_list.csv')

#define combin saving function
def save_combin(combination):
    # Generate new combin_id
    new_id = st.session_state.save_list['combin_id'].max() + 1 if not st.session_state.save_list.empty else 1
    new_id_row = pd.DataFrame({'combin_id': [new_id]})

    # Concatenate new_id_row to save_list
    st.session_state.save_list = pd.concat([st.session_state.save_list, new_id_row], axis=0, ignore_index=True)

    # Save save_list to CSV
    st.session_state.save_list.to_csv('my library/save_list.csv', index=False)

    # Save combination to CSV
    combination.to_csv(f'my library/combins{new_id}.csv', index=False)

#define combin file loading function
def load_saved_combins(filename):
    return pd.read_csv(filename)

# for my wardrobe page
def show_wardrobe():
    st.title("My Wardrobe")
    num_items = len(st.session_state.wardrobe_df) 
    num_rows = (num_items + 2) // 3  # Ensure we have enough rows for all items

    for i in range(num_rows):
        with st.container():
            row_columns = st.columns(3)  # Create three columns for each row
            for j in range(3):
                item_index = i * 3 + j
                if item_index < num_items:
                    item = st.session_state.wardrobe_df.iloc[item_index]
                    with row_columns[j]:
                        with st.container(border=True):
                            style_button()
                            col1,col2, col3 = st.columns([1,3,1])
                            centered_text = f"Type: {item['type']}"
                            centered_color = f"Color: {item['color']}"
                            col2.markdown(f"<center>{centered_text}</center>", unsafe_allow_html=True)
                            col2.markdown(f"<center>{centered_color}</center>", unsafe_allow_html=True)
                            if col3.button('❌', key=f"Delete {item_index}"):
                                st.session_state.wardrobe_df.drop(st.session_state.wardrobe_df.index[item_index], inplace=True)
                                st.session_state.wardrobe_df.reset_index(drop=True, inplace=True)
                                st.session_state.wardrobe_df.to_csv('wardrobe.csv', index=False)
                                st.rerun()
                            st.image(str(item['image_link']) + '.jpg', width=150, use_column_width=True)

                            

# for pair finder page
                     
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None

def show_pair_finder():
    if st.session_state.selected_item is not None:
        col1, col2, col3 = st.columns([3, 4, 1])  # Adjust column widths as needed
        col1.title("Pair Finder")
        col2.empty()
        col3.write("") 
        col3.write("")  
        if col3.button("back"):
            st.session_state.selected_item = None
            st.rerun()
        
        spacing()
        spacing()

        with st.container(border=True):
            col1, col2 = st.columns(2)
        
            with col1:
                st.image(st.session_state.selected_item['image_link'] + '.jpg', width=150, use_column_width=True)
            
            with col2.container(border=True):
                centered_text = f"Type: {st.session_state.selected_item['type']}"
                centered_color = f"Color: {st.session_state.selected_item['color']}"
                st.markdown(f"<center>{centered_text}</center>", unsafe_allow_html=True)
                st.markdown(f"<center>{centered_color}</center>", unsafe_allow_html=True)

        st.subheader("Pairs")
        with st.spinner('Finding best pairs...'):
            matches = find_pairs(st.session_state.selected_item,st.session_state.wardrobe_df)

        for combi_type in matches['combi_type'].unique():
            filtered_matches = matches[matches['combi_type'] == combi_type].sort_values(by='predicted_match', ascending=False)[:3]
            

            with st.expander(combi_type):

                num_items = len(filtered_matches) 
                num_rows = (num_items + 2) // 3  # Ensure we have enough rows for all items

                for i in range(num_rows):
                    with st.container():
                        row_columns = st.columns(3)  # Create three columns for each row
                        for j in range(3):
                            item_index = i * 3 + j
                            if item_index < num_items:
                                item = filtered_matches.iloc[item_index]
                                with row_columns[j]:
                                    with st.container(border=True):
                                        centered_text = f"Type: {item['combi_type']}"
                                        centered_color = f"Color: {item['combi_color']}"
                                        st.markdown(f"<center>{centered_text}</center>", unsafe_allow_html=True)
                                        st.markdown(f"<center>{centered_color}</center>", unsafe_allow_html=True)
                                        st.image(item['combi_image_link'] + '.jpg', width=150,use_column_width=True)  

    else:
        st.title("Pair Finder")

        types = st.session_state.wardrobe_df['type'].unique().tolist()  # Add an empty string for no initial selection
        
        selected_index = None

        if 'selected_type' in st.session_state and st.session_state.selected_type is not None:
            if st.session_state.selected_type in types:
                selected_index = types.index(st.session_state.selected_type)    

        selected_type = st.selectbox("Select Type", types, index=selected_index)

        st.session_state.backup = selected_type

        if selected_type:  # Check if a type is selected (not an empty string)
            # Filter wardrobe DataFrame by selected type
            filtered_items = st.session_state.wardrobe_df[st.session_state.wardrobe_df['type'] == selected_type]

            num_items = len(filtered_items)
            num_rows = (num_items + 2) // 3  # Ensure we have enough rows for all items

            for i in range(num_rows):
                with st.container():
                    row_columns = st.columns(3)  # Create three columns for each row
                    for j in range(3):
                        item_index = i * 3 + j
                        if item_index < num_items:
                            item = filtered_items.iloc[item_index]
                            with row_columns[j]:
                                with st.container(border=True):
                                    centered_text = f"Type: {item['type']}"
                                    centered_color = f"Color: {item['color']}"
                                    st.markdown(f"<center>{centered_text}</center>", unsafe_allow_html=True)
                                    st.markdown(f"<center>{centered_color}</center>", unsafe_allow_html=True)
                                    st.image(item['image_link'] + '.jpg', width=150, use_column_width=True)
                                    if st.button("Choose",key=item_index, use_container_width=True):
                                        st.session_state.selected_item = item
                                        st.session_state.selected_type = selected_type
                                        st.rerun() 

#for combin of day! page
def show_combins():
    st.title("Combins of Day")
    if 'combins' in st.session_state and st.session_state.combins is not None:
        combins = st.session_state.combins
        num_items = len(combins) 
        num_rows = (num_items + 2) // 3  

        for i in range(num_rows):
            with st.container():
                row_columns = st.columns(3)  
                for j in range(3):
                    item_index = i * 3 + j
                    if item_index < num_items:
                        item = combins.iloc[item_index]
                        with row_columns[j]:
                            with st.container(border=True):
                                centered_text = f"Type: {item['combi_type']}"
                                centered_color = f"Color: {item['combi_color']}"
                                st.markdown(f"<center>{centered_text}</center>", unsafe_allow_html=True)
                                st.markdown(f"<center>{centered_color}</center>", unsafe_allow_html=True)
                                st.image(item['combi_image_link'] + '.jpg', width=150,use_column_width=True)

        # Add "Save Combin" and "Find New Combin" buttons                       
        button_col1, button_col2,button_col3,button_col4 = st.columns(4)
        if button_col1.button('**Save Combin ❤️**'):
            save_combin(st.session_state.combins)
            st.success('Combin saved successfully!')

        if button_col4.button('**Find New Combin**'):
            st.session_state.combins = None
            st.session_state.retry = True
            st.rerun()
    elif 'retry' in st.session_state and st.session_state.retry is not None: 
        with st.spinner("Preparing a combin..."):
            st.session_state.combins = make_combins_day()
            st.rerun()
    else: 
        if st.button('Find me a combin'):
            with st.spinner("Preparing a combin..."):
                st.session_state.combins = make_combins_day()
                st.rerun()

# for add to wardrobe page
def show_add_wardrobe():
    st.title("Add to Wardrobe")
    with st.form(key='image_form'):
        # Upload image
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        # Select type
        selected_type = st.selectbox("Select Type", types,index=None)

        # Select color
        selected_color = st.selectbox("Select Color", colors,index=None)

        # Submit button
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            if uploaded_image is not None and selected_color is not None and selected_type is not None:
                image_bytes = uploaded_image.read()  # Get the bytes of the uploaded image
                image = Image.open(BytesIO(image_bytes))  # Open image using PIL
                image_name = os.path.splitext(uploaded_image.name)[0]  # Remove file extension from image name
                image_path = f"streamlit_images/{image_name}"  # Define path to save the image
                image.save(image_path + '.jpg')  # Save image to disk

                new_item = {
                    'image_link': image_path,
                    'type': selected_type,
                    'color': selected_color
                }
                st.session_state.wardrobe_df = st.session_state.wardrobe_df.append(new_item, ignore_index=True)
                st.session_state.wardrobe_df = st.session_state.wardrobe_df.sort_values(by='type')
                st.session_state.wardrobe_df.to_csv('wardrobe.csv')

                st.success("Item added to wardrobe successfully!")
                st.image(uploaded_image, use_column_width=True)
                st.write("Type:", selected_type)
                st.write("Color:", selected_color)
            else:
                st.write('Please fill all fields')

#for my library page
def show_library():
    st.title("My Library")

    for _, row in st.session_state.save_list.iterrows():
        id = row.loc['combin_id']  # Get combin_id from the row
        
        # Load saved_combins from CSV
        try:
            saved_combins = load_saved_combins(f'my library/combins{id}.csv')
        except FileNotFoundError:
            st.warning(f"Combination with ID {id} not found.")
            continue  # Skip to the next combination if file not found
        
        # Display saved_combins in a container
        with st.container(border=True):
            col1,col2,col3=st.columns([6,5,1])
            col1.subheader("Combin:")

            if col3.button('❌', key=f"Delete_{id}"):
                st.session_state.save_list = st.session_state.save_list[st.session_state.save_list['combin_id'] != id]
                st.session_state.save_list.to_csv('my library/save_list.csv', index=False)
                st.rerun()

            num_items = len(saved_combins) 
            num_rows = (num_items + 3) // 4  # Ensure we have enough rows for all items
 
            for i in range(num_rows):
                with st.container():
                    row_columns = st.columns(4)  # Create four columns for each row
                    for j in range(4):
                        item_index = i * 4 + j
                        if item_index < num_items:
                            item = saved_combins.iloc[item_index]
                            with row_columns[j]:
                                with st.container(border=True):
                                    centered_text = f"Type: {item['combi_type']}"
                                    centered_color = f"Color: {item['combi_color']}"
                                    st.markdown(f"<center>{centered_text}</center>", unsafe_allow_html=True)
                                    st.markdown(f"<center>{centered_color}</center>", unsafe_allow_html=True)
                                    st.image(item['combi_image_link'] + '.jpg', width=150, use_column_width=True)


#for home page
def show_welcome():
    st.image("name.png", use_column_width=True)
    st.image("Women-Clothes-Icons.jpg", use_column_width=True)
    

# Initial page number assignment                
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0 

# to remove spacing
def main():
    st.markdown("""
        <style>
               .block-container {
                    margin-top: -6rem;
                }
        </style>
        """, unsafe_allow_html=True)

# left sidebar buttons
    with st.sidebar:
        spacing()
        spacing()
        spacing()
        if st.button("**:grey[My Wardrobe]**", use_container_width=True):
            st.session_state.page_number = 1
        spacing() 
        spacing()
        if st.button('**:grey[Add to Wardrobe]**', use_container_width=True):
            st.session_state.page_number = 2  
        spacing()
        spacing() 
        if st.button('**:grey[Pair Finder]**', use_container_width=True):
            st.session_state.page_number = 3
        spacing()
        spacing()
        if st.button('**:grey[Combin of day!]**', use_container_width=True):
            st.session_state.page_number = 4
        spacing()
        spacing()
        if st.button('**:grey[My Library]**', use_container_width=True):
            st.session_state.page_number = 5
        spacing()
        spacing()
        spacing()
        st.image("logo.jpeg", use_column_width=True)


 # buttons order       
    if st.session_state.page_number == 0:
        show_welcome()
    elif st.session_state.page_number == 1:
        st.session_state.selected_type = st.session_state.backup
        show_wardrobe()

    elif st.session_state.page_number == 2:
        st.session_state.selected_type = st.session_state.backup
        show_add_wardrobe()

    elif st.session_state.page_number == 3:
        show_pair_finder()

    elif st.session_state.page_number == 4:
        st.session_state.selected_type = st.session_state.backup
        show_combins()

    elif st.session_state.page_number == 5:
        st.session_state.selected_type = st.session_state.backup
        show_library()
    


if __name__ == "__main__":
    main()
