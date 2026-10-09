#include <cstring>

#include <iostream>

#include <algorithm>



using namespace std;



int N, A, B, rings[100], G[100][100]; bool used[100];



int rec(int pos)

{

	int ret = 1; used[pos] = true;



	for(int i = 0; i < rings[pos]; i++)

	{

		int r = G[pos][i];

		

		if(!used[r])

		{

			ret = max(ret, rec(r) + 1);

		}

	}



	used[pos] = false;



	return ret;

}



int main()

{

	int N, A, B;



	while(true)

	{

		scanf("%d", &N);

 

		if(N == 0) { break; }



		for(int i = 0; i < 100; i++)

		{

			used[i] = false;

			rings[i] = 0;

		}



		for(int i = 0; i < N; i++)

		{

			scanf("%d", &A); A--;

			scanf("%d", &B); B--;



			G[A][rings[A]] = B; rings[A]++;

			G[B][rings[B]] = A; rings[B]++;

		}



		int nodes = 0;



		for(int i = 0; i < 100; i++)

		{

			if(rings[i])

			{

				nodes++;

			}

		}



		int ret = 0;



		for(int i = 0; i < 100; i++)

		{

			ret = max(ret, rec(i));



 			if(ret == nodes) break;

		}

 

		printf("%d\n", ret);

	}



	return 0;

}