#include<iostream>

#include<cstring>

using namespace std;

int main()

{

	int k;

	string s;

	cin>>k>>s;

	if(s.length()>k)

	{

		for(int i=0;i<k;i++)

		cout<<s[i];

		cout<<"...";

	}

	else

	cout<<s;

} 