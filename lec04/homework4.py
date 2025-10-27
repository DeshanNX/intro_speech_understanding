
def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    # Convert all keys (month, day) to a sorted list
    sorted_dates = sorted(birthdays.keys())

    # Loop through sorted dates and find the first one that comes after the given date
    for bday in sorted_dates:
        if bday > date:     # tuple comparison works automatically in Python
            return bday, birthdays[bday]

    # If no later date is found, wrap around to the first in the year
    first_date = sorted_dates[0]
    return first_date, birthdays[first_date]
