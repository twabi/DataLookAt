import ntplib
from datetime import datetime, timezone
#c = ntplib.NTPClient()
# Provide the respective ntp server ip in below function
#response = c.request('133.41.117.50', version=3)

#print(response.tx_time + response.offset)
#print (datetime.fromtimestamp(response.tx_time, timezone.utc))

# Create a datetime object for JST timezone
#jst = datetime.timezone(datetime.timedelta(hours=9))
dt = datetime.now()

print(dt)

# Convert datetime object to Unix timestamp
unixtime = dt.timestamp() * 1000

print(f"Unix timestamp: {unixtime}")
