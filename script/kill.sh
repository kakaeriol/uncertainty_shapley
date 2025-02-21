for i in {1..1000..10}
do
	`kill $(ps -ef | grep main.py | awk '{print $2}')`
	sleep 3
done
