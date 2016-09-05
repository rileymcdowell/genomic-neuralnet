from __future__ import print_function

import boto3

def get_master_dns(public=False):
    ec2 = boto3.resource('ec2', region_name='us-east-1') 
    instances = ec2.instances.all()
    master = None
    for instance in instances:
        tags = instance.tags
        state = instance.state['Name']
        for t in tags:
            if state == 'running' and t['Key'] == 'QueueRole' and t['Value'] == 'Master':
                master = instance

    if master == None:
        print('WARNING: No AWS Queue instance detected. Falling back to localhost')
        return 'localhost' # AWS master instance not tagged.
    if public:
        return master.public_dns_name
    else:
        return master.private_dns_name


def get_slave_dns_list():
    ec2 = boto3.resource('ec2') 
    instances = ec2.instances.all()
    slaves = []
    for instance in instances:
        tags = instance.tags
        for t in tags:
            if t['Key'] == 'QueueRole' and t['Value'] == 'Slave':
                slaves.append(instance) 

    return [i.private_dns_name for i in slaves]

def main():
    print(get_master_dns())
    print(get_slave_dns_list())

if __name__ == '__main__':
    main()

