import os
import sys
import datetime
import json
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType

def load_config(config_path=None):
    """Load configuration from a JSON file or environment variables"""
    config = {}
    
    # Try to load from config file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
    
    # Environment variables override file config
    config['jdbc_url'] = os.environ.get('JDBC_URL', config.get('jdbc_url', 'jdbc:postgresql://postgres:5432/postgres'))
    config['table_name'] = os.environ.get('TABLE_NAME', config.get('table_name', 'public.sales'))
    config['db_user'] = os.environ.get('DB_USER', config.get('db_user', 'postgres'))
    config['db_password'] = os.environ.get('DB_PASSWORD', config.get('db_password', 'postgres'))
    config['primary_key'] = os.environ.get('PRIMARY_KEY', config.get('primary_key', 'salesid'))
    
    config['s3_bucket'] = os.environ.get('S3_BUCKET', config.get('s3_bucket', 'my-data-bucket'))
    config['s3_prefix'] = os.environ.get('S3_PREFIX', config.get('s3_prefix', 'sales-data'))
    config['file_format'] = os.environ.get('FILE_FORMAT', config.get('file_format', 'parquet'))
    config['checkpoint_dir'] = os.environ.get('CHECKPOINT_DIR', config.get('checkpoint_dir', './checkpoint'))
    
    config['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID', config.get('aws_access_key_id', ''))
    config['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY', config.get('aws_secret_access_key', ''))
    config['s3_endpoint'] = os.environ.get('S3_ENDPOINT', config.get('s3_endpoint', 's3.amazonaws.com'))
    
    return config

def main():
    # Get configuration file path from command line argument, if provided
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)
    
    # Set up PySpark with PostgreSQL driver
    SUBMIT_ARGS = "--packages org.postgresql:postgresql:42.5.4,org.apache.hadoop:hadoop-aws:3.3.1 pyspark-shell"
    os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    # Create Spark session with S3 configuration
    spark = SparkSession.builder \
        .appName('IncrementalJDBC') \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.access.key", config['aws_access_key_id']) \
        .config("spark.hadoop.fs.s3a.secret.key", config['aws_secret_access_key']) \
        .config("spark.hadoop.fs.s3a.endpoint", config['s3_endpoint']) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .getOrCreate()
    
    # Print configuration (excluding sensitive data)
    print("Running with configuration:")
    print(f"JDBC URL: {config['jdbc_url']}")
    print(f"Table: {config['table_name']}")
    print(f"Primary Key: {config['primary_key']}")
    print(f"S3 Bucket: {config['s3_bucket']}")
    print(f"S3 Prefix: {config['s3_prefix']}")
    print(f"File Format: {config['file_format']}")
    print(f"Checkpoint Directory: {config['checkpoint_dir']}")
    print(f"S3 Endpoint: {config['s3_endpoint']}")
    
    # Ensure checkpoint directory exists
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Path for max_id checkpoint
    max_id_checkpoint_dir = f"{config['checkpoint_dir']}/max_id"
    
    # Read the maximum value of the primary key from the last extraction
    if os.path.exists(max_id_checkpoint_dir):
        try:
            max_id = spark.read.csv(max_id_checkpoint_dir).collect()[0][0]
            print(f"Checkpoint found, resuming from {max_id}")
        except Exception as e:
            print(f"Error reading checkpoint: {str(e)}, starting from scratch.")
            max_id = 0
    else:
        print("Checkpoint not found, starting from scratch.")
        max_id = 0
    
    # Build the incremental query
    incremental_query = f"SELECT * FROM {config['table_name']} WHERE {config['primary_key']} > {max_id}"
    
    # Read the incremental data from the database
    try:
        print(f"Executing query: {incremental_query}")
        incremental_data = spark.read.format('jdbc').options(
            url=config['jdbc_url'],
            query=incremental_query,
            user=config['db_user'],
            password=config['db_password'],
            driver='org.postgresql.Driver'
        ).load()
        
        # Process only if there's new data
        record_count = incremental_data.count()
        print(f"Found {record_count} new records")
        
        if record_count > 0:
            # Get the timestamp for the filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Extract table name without schema for filename
            table_name_only = config['table_name'].split('.')[-1]
            
            # Define S3 path
            s3_path = f"s3a://{config['s3_bucket']}/{config['s3_prefix']}/{table_name_only}_{timestamp}"
            print(f"Writing data to {s3_path}")
            
            # Save data to S3 based on format
            if config['file_format'].lower() == 'parquet':
                incremental_data.write.mode("overwrite").parquet(s3_path)
                print(f"Data saved as parquet to {s3_path}")
            else:  # default to CSV
                incremental_data.write.mode("overwrite").option("header", "true").csv(s3_path)
                print(f"Data saved as CSV to {s3_path}")
            
            # Update the checkpoint with the new max_id
            new_max_id = incremental_data.agg({config['primary_key']: "max"}).collect()[0][0]
            spark_df = spark.createDataFrame(data=[(str(new_max_id), config['table_name'])], 
                                             schema=['max_id', "table_name"])
            print(f"Updating checkpoint with max_id: {new_max_id}")
            spark_df.write.mode("overwrite").csv(max_id_checkpoint_dir)
            
            # Print sample of data
            print("Sample of fetched data:")
            incremental_data.show(5)
            print(f"Successfully processed {record_count} new records")
        else:
            print("No new data to process")
    
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close Spark session
        spark.stop()
        print("Spark session closed")

if __name__ == "__main__":
    main()