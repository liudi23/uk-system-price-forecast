#create a function to calculate business days
from datetime import datetime, timedelta
def calculate_business_days(start_date, end_date):
    business_days = 0
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday are considered business days
            business_days += 1
        current_date += timedelta(days=1)
    return business_days
# Example usage
start_date = datetime(2024, 6, 1)  # June 1
end_date = datetime(2024, 6, 30)   # June 30
business_days_count = calculate_business_days(start_date, end_date)
print(f"Number of business days between {start_date.date()} and {end_date.date()}: {business_days_count}")
