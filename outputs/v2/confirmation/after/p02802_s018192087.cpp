#include<bits/stdc++.h>

using namespace std;

int main()

{

	char a[10];

	int c[100005]={0},d[100005]={0};

	int m,n,b,x=0,y=0,z=0,i;

	scanf("%d%d",&m,&n);

	for(i=1;i<=n;i++)

	{

		scanf("%d%s",&b,a);

		if(d[b]==-1)

		continue;

		if(a[0]=='A')

		{

			d[b]=-1;

			x++;

			z+=c[b];

			//printf("%d ",z);	

		} 

		if(a[0]=='W')

		{

			c[b]++;

		}

	}

	if(n==0)

	printf("0 0");

	else

	printf("%d %d",x,z);

}