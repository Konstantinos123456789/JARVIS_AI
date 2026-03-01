import datetime
from help import speak, take_user_input

# You can customize these special days
special_days = {
    (12, 25): "Christmas",
    (1, 1): "New Year",
    
}

def check_special_days():
    """Check for upcoming special days in the next 6 years"""
    today = datetime.date.today()
    announced = set()  # Keep track of what we've announced
    
    for (month, day), description in special_days.items():
        for year in range(today.year, today.year + 6):
            try:
                special_day = datetime.date(year, month, day)
                if special_day >= today and (month, day) not in announced:
                    days_until = (special_day - today).days
                    if days_until == 0:
                        speak(f"Today is {description}!")
                    elif days_until <= 30:  # Only announce if within 30 days
                        speak(f"{special_day.strftime('%B %d, %Y')} is {description}, in {days_until} days!")
                    announced.add((month, day))
                    break  # Only announce the next occurrence
            except ValueError:
                # Invalid date (like February 30)
                continue

def add_special_day():
    """Add a new special day to the calendar"""
    while True:
        try:
            speak("What is the month of the special day? Please say the month number, for example, 12 for December.")
            month_input = take_user_input()
            if month_input == 'None':
                speak("I didn't catch that. Let me try again.")
                continue
            
            # Try to parse the month
            try:
                month = int(month_input.split()[0])  # Get first word/number
            except (ValueError, IndexError):
                speak("I need a number for the month. Let me try again.")
                continue
                
            if month < 1 or month > 12:
                speak("Invalid month. Please use a number between 1 and 12.")
                continue
            
            speak("What is the day? Please say the day number.")
            day_input = take_user_input()
            if day_input == 'None':
                speak("I didn't catch that. Let me try again.")
                continue
            
            try:
                day = int(day_input.split()[0])  # Get first word/number
            except (ValueError, IndexError):
                speak("I need a number for the day. Let me try again.")
                continue
                
            if day < 1 or day > 31:
                speak("Invalid day. Please use a number between 1 and 31.")
                continue
            
            # Validate the date
            try:
                datetime.date(2024, month, day)  # Use 2024 to check if date is valid
            except ValueError:
                speak(f"Invalid date. Month {month} doesn't have {day} days.")
                continue
            
            speak("What is the description of this special day?")
            description = take_user_input()
            if description == 'None':
                speak("I didn't catch that. Let me try again.")
                continue
            
            special_days[(month, day)] = description
            speak(f"Special day added: {description} on {month}/{day}!")
            break
            
        except Exception as e:
            print(f"Error adding special day: {e}")
            speak("Sorry, there was an error. Let me try again.")
