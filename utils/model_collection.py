# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:39:45 2021

@author: Daniel Sperber
"""

import tensorflow as tf
import pandas as pd
import numpy as np

import hashlib


# globals

HASH_COLUMN_NAME = 'Hash value for saving'

Folder_Name = r"SavedModels" # assuming this is not ""
File_Name = "model_collection.csv"

sep = '\t'

# created in _add_df_to_collection
Active_Collection = None 


def _get_full_path():
    return Folder_Name + "/" + File_Name 


def __getattr__(key):
    if key in ("mc", "collection"):
        return get_collection()
    raise AttributeError("module " + __name__ + " has no attribute '"+key+"'") # small todo, co


####

def get_collection(path=None, reload=False):
    global Active_Collection
    if Active_Collection is not None and not reload:
        return Active_Collection
    if path is None:
        path = _get_full_path()
    try:
        # Check for collection csv
        Active_Collection = pd.read_table(path, sep=sep, index_col=0)
        return Active_Collection
    except FileNotFoundError as e:
        #raise 
        print(FileNotFoundError("WARNING: No active collection, and " + str(e)))
        return None
             
####

def get_model_names(tolist=True):
    all_models = get_collection()
    if tolist:
        return all_models['Name'].tolist()
    return all_models['Name']    

####

def model_get_value(name, value_name):
    return get_collection()[value_name][get_collection().Name == name].item()
    

def get_model_hash(model_settings):
    # Note that list and tuples containing the same items will yield
    # different hashes
    m = hashlib.sha256()
    
    # As dicts are sorted in the newer python versions
    # {'a': 1, 'b': 2} is in order not the same as {'b': 2, 'a': 1}
    # and will yield different hashes, therefore using sorted
    
    for k in sorted(model_settings.keys()):
        # As i might change from current ugly uppercase to something different
        m.update(bytes(k.lower(), 'utf-8'))
        val = model_settings[k]
        if type(val) == bool: # Treat True as 1 alike
            val = int(val)
        m.update(bytes(str(model_settings[k]), 'utf-8')) 
    return m.hexdigest()


def update_model_hash(index, **new_settings):
    """
    In case a new option is added which are not present for existing models.
    This updates their hash and data with new_setting=default value.
    
    Multiple keywords=values can be passed as keyword arguments.
    
    The new_setting column must be added present.
    
    All columns, expect Name, before the hash column will be used to generate
    the new hash. Which should match the new settings
    """
    # TODO unfinished
    collection = get_collection()
    data = collection.iloc[index]
    cols = data.columns
    end_idx = cols.tolist().index('Hash value for saving')
    mask = range(1, end_idx)
    
    

####

def is_model_in_collection(model_settings):
   if HASH_COLUMN_NAME in model_settings:
       hash_val = model_settings[HASH_COLUMN_NAME]
   else:
       hash_val = get_model_hash(model_settings)
   
   collection = get_collection()
   if collection is None:
       return None
   return hash_val in collection[HASH_COLUMN_NAME].values


# Checks for collection and index
def get_model_index_from_hash(hash_val):
    collection = get_collection()
    if collection is None:
        return None
    # sigh why is there no nice numpy function for getting first occurence?
    model_df_index = np.argmax(collection[HASH_COLUMN_NAME].values == hash_val)
    if model_df_index > 0:
        return model_df_index
     # argmax and non existing return 0
    if collection[HASH_COLUMN_NAME].values[0] == hash_val:
        return 0
    return None


###

# Creates a new csv file for every model... and overrittes
def _add_df_to_collection(settings_df, hash_val, name="", save_as_csv=False, allow_duplicates=False):
    global Active_Collection
    try:
        collection = get_collection()
    except FileNotFoundError:
        collection = None

    # Create collection if .csv file does not exist
    if collection is None:
        print(_get_full_path(), "not found. Creating new database")
        # Use passed df as new
        collection = Active_Collection = settings_df

        if 'Name' not in collection.keys():
            collection.insert(loc=0, column='Name', value=name)
        
        print("Collection:")
        print(collection)
        
        # Insert name
        model_exists = False # Assuming no db = no model
        model_df_index = 0        
    else:
        # Standard case
        # check if hash value exist, then don't insert
        model_df_index = get_model_index_from_hash(hash_val)
        if model_df_index is not None:
            saved_name = collection['Name'][model_df_index]
            print("\nModel has been already saved with hash", hash_val, 
                  "\nName(index):", saved_name, f"({model_df_index})")
            # maybe has been deleted
            print("Checking if keras model is still present...")
            try:
                # Could do this probably nicer with just os but this is safe.
                tf.keras.models.load_model(Folder_Name + "/" + saved_name, compile=False)
                model_exists = True
            except (OSError, ImportError):
                print("Model no longer exists.")
                model_exists = False
            if allow_duplicates:
                collection = Active_Collection = collection.append(settings_df, ignore_index=True)
                # Assume it does not exist for further entries.
                model_exists = False
        
        else:
            print("Model is not in the collection. Adding it")
            model_exists = False
            model_df_index = len(collection) # actually -1 but new row has not been added yet
            # Adding model to df, ignore index, allows adding new features
            if 'Name' not in settings_df.keys():
                settings_df.insert(loc=0, column='Name', value=name)

            collection = Active_Collection = collection.append(settings_df, ignore_index=True)

    #finally:
    # Saving is optional maybe further data shall be added.
    if save_as_csv:
        save_collection_as_csv(collection)
        #collection.to_csv(_get_full_path(), sep=sep)

    return model_exists, collection, model_df_index

def add_model_to_collection(model, model_settings, name="Not named", save_model=True, overwrite=True, save_collection=True):
    """
    Saves the model to the folder declared by model_collection.Folder_Name and the passed name.
    An entry will be created in the .csv file named model_collection.File_Name for management.
    
    A pandas DataFrame of this .csv file will be returned to store further data.
    You can save the DataFrame with save_collection_as_csv(DataFrame).
    """
    hash_val = get_model_hash(model_settings)
    # save hash value
    model_settings[HASH_COLUMN_NAME] = hash_val
    
    # Must be turned into a list for 1D DF, not via list(dict) !
    df = pd.DataFrame([model_settings])
    model_exists, collection, model_df_index = _add_df_to_collection(df, hash_val, name, save_collection, allow_duplicates=overwrite)
    if save_model and (not model_exists or overwrite):
        if model_exists:
            print("Overwritting model.")
            model_exits = False #todo maybe small fix to save data on double names
        model.save(Folder_Name + "/" + name)
        # Tensorflow prints feedback
        # print("Saved model at", Folder_Name + "/" + name) 
    else:
        print("Model already exist or saving is disabled. Not saving trained model")
                
    
    return model_exists, collection, model_df_index
    
###

def save_collection_as_csv(collection_df=None):
    # function only exist for consitency
    # If collection was adjusted from the outside
    # As for example by adding data and maybe copying it.
    if collection_df is None:
        collection_df = get_collection()
    #NOTE: Maybe keep index to load index instead of name / if same name is there multiple times.
    while True:
        try:
            collection_df.to_csv(_get_full_path(), sep=sep, index=range(len(collection_df)))
            break
        except PermissionError as e:
            print(e)
            input("Close file and press any key.")
        
        
save = save_collection_as_csv
     
# =============================================================================
# Loading Models
# =============================================================================
    
# Automatically by hash
def check_for_saved_model(model_settings):
    """
    Returns None if not found or couldn't load.
    Else the model and the index in the Active_Collection.
    
    Index will be returned by add_model_to_collection
    """
    hash_val = get_model_hash(model_settings)
    
    print("Hash value is", hash_val)
    
    collection_index = get_model_index_from_hash(hash_val)
    collection = get_collection()
    if collection is None:
        print("No collection found at:", Folder_Name + "\\" + File_Name)
    
    if collection_index is not None and collection is not None:
        try:
            print("Loading model with index", collection_index, "and name", collection['Name'][collection_index])
            model = tf.keras.models.load_model(Folder_Name + "/" + collection['Name'][collection_index])
            return True, Active_Collection, collection_index, model
        except (OSError, ImportError) as e:
            print("Warning: Model was once saved but coudn't load it.\n", e)
            pass
    return False, Active_Collection, collection_index, None

# manually

def _load_model_by_name(name, collection_index, folder=""):
    # If we have the name we can load directly
    if bool(folder) == False:
        folder = Folder_Name
    try:
        print("Loading model with name:", name)
        model = tf.keras.models.load_model(folder + "/" + name)
        return True, Active_Collection, collection_index, model
    except (OSError, ImportError) as e:
        print("Warning: No such model in folder:", folder, e)
        pass
    return False, Active_Collection, collection_index, None


def load_model_by_name(name, folder=""):
    """
    Name as seen in the csv file.
    """
    collection = get_collection()
    if name not in collection['Name'].values:
        print(name, "is not in collection, will try to load it anyway.")
        collection_index = None
    else:
        collection_index = collection.Name[collection.Name == name].index[0]
    
    return _load_model_by_name(name, collection_index, folder)
    
    
def load_model_by_index(idx, folder=""):
    """
    Index as seen in the csv file.
    """
    collection = get_collection()
    name = collection.loc[idx, 'Name']
    return _load_model_by_name(name, idx, folder)

#

def get_model_settings(idx, path=None):
    """
    idx can be a name or integer for indexing
    """
    collection = get_collection(path)
    # could do this without a copy direct with loc / iloc
    if type(idx) == str:
        data = collection.set_index('Name').loc[idx]
    else:
        data = collection.loc[idx]
    cols = collection.columns.tolist()
    # Between these columns are the settings
    startidx = cols.index('Topic') # either 1 or 2 depending on type]
    endidx = cols.index('Total params')
    data = data.iloc[startidx:endidx]
    
    # Cast data to correct type as they come as string
    data = data.to_dict()
    for k,v in data.items():
        try:
            data[k] = eval(v)
        except:
            # value is a string and eval fails
            continue
    
    return data
    

# =============================================================================
# Adding further data
# =============================================================================

# combine should work, not using this anymore
def _collection_make_mergeable(collection_df, update_df):
    if len(collection_df.columns) != len(update_df.columns):
       new_cols = [col for col in update_df.columns if col not in collection_df.columns]
       # NOTE Assuming the last two columns are ALWAYS date and PDFs used
       for i, col in enumerate(new_cols, start=len(collection_df.columns) - 2):
           collection_df.insert(loc=i, column=col, value=pd.NA)
    elif np.any(collection_df.columns != update_df.columns):
        # should not happend
        print("WARNING: collection csv and updated row have same length but different columns")



def collection_update_row(collection_df, index, update_data, update_Active_Collection=True):
    #_collection_make_mergeable(collection_df, update_df) # replaced with combine, order is probably lost.
    
    if not isinstance(update_data, pd.core.frame.DataFrame):
        # should come as a dict
        update_data = pd.DataFrame([update_data], index=[index])
    
    new_pdf = update_data['PDF'][index] # maybe, "", None
    if pd.notna(new_pdf) and bool(new_pdf):
        if 'PDF' in collection_df.columns:
            # This is normaly the case, only not if gernerating a new one
            print("PDF value is", collection_df.loc[index, 'PDF'], "NA?", pd.isna(collection_df.loc[index, 'PDF']))
            if pd.notna(collection_df['PDF'][index]):
                # model was used in another PDF
                old_pdfs = collection_df.loc[index, 'PDF']
                # Check if PDF is already present
    
                # maybe pdf is already present
                if new_pdf not in old_pdfs: 
                    # if not found add it
                    collection_df.loc[index, 'PDF'] = old_pdfs + "\n" + new_pdf
            else:
                collection_df.loc[index, 'PDF'] = new_pdf # Use loc when setting items!
        else:
            # When creating a new .csv file, skipping 
            print("Column PDF not present in .csv or data Frame")
            collection_df.insert(len(collection_df), 'PDF', new_pdf)
        
    del update_data['PDF']
    
    # To restore order after combine and remove doubles, works while dicts are ordered
    cols = list(dict.fromkeys(collection_df.columns.to_list() + update_data.columns.to_list()))


    # Combine first is the most flexible here
    # but casts values to float so changing type...
    collection_df = collection_df.astype(object)
    # this it not inplace
    collection_df = update_data.combine_first(collection_df)

    # move PDF to the end
    cols.remove('PDF')
    cols.append('PDF')
    collection_df = collection_df.reindex(columns=cols) # not in place
    
    if update_Active_Collection:
        global Active_Collection
        Active_Collection = collection_df
        
    return collection_df
    
# =============================================================================
# Fixdata
# =============================================================================



def fix_collection():
    """
    Working on the .csv with a spreadsheet programm can dislocate the data types.
    This function needs to be updated with every significant change.
    """
    df = get_collection()
    # Fix string bools
    df.replace('WAHR', True, inplace=True)
    df.replace('TRUE', True, inplace=True)
    df.replace('FALSCH', False, inplace=True)
    df.replace('FALSE', False, inplace=True)
    #Fix 1,23 and 1,23e-4
    for col in df:
        if col not in ["Name", "PDF", "Topic", "Hash value for saving", "Environment", "Comment1", "Comment2", "Average time (epoch)", "Total time"]:
            if df[col].dtype == object:
                mask = df[col].str.match("^\d").fillna(False)
                df.loc[mask, col] = df.loc[mask, col].replace(",", ".", regex=True).astype(float)
                
    return df
                
                
    