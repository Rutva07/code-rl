#include<stdio.h>

#include<string.h>



int main()

{

	int n;

	scanf("%d",&n);

	long long int ans=1;

	for(int i=1;i<=n;i++) ans=(ans*i)%1000000007;

	printf("%lld\n",ans);

	return 0;

}