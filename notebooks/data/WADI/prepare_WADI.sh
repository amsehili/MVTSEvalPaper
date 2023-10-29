#/bin/bash!

echo "Preparing WADI 2017 dataset...";

sed -i 1,4d WADI_14days.csv;
sed -i 's/\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\//g' WADI_14days.csv;
sed -i 's/\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\//g' WADI_attackdata.csv;

echo "done!";

