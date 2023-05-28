#include <iostream>
#include <iomanip>
#include <vector>
#include <omp.h>

using namespace std;

void sequential_bubble_sort(vector<int>& arr){
    vector<int> array = arr;
        
    // Measure performance of sequential bubble sort
    double start = omp_get_wtime();
    
    for(int i = 0; i < array.size() - 1; i ++){
        for(int j = 0; j < array.size() - i - 1; j++){
            if(array[j] > array[j+1]){
                swap(array[j],array[j+1]);
            }
        }
    }
    
    double end = omp_get_wtime();
    cout << "After Sequential Bubble Sort: ";
    for(int i = 0; i < array.size(); i++)
    	cout << array[i] << " ";
    cout << endl;
  
    cout << "Time Required using Sequential Bubble Sort: " <<setprecision(6)<< end - start << endl;

}

void bubble_sort_odd_even(vector<int>& arr) {
    vector<int> array = arr;

    // Measure performance of parallel bubble sort using odd-even transposition
    double start = omp_get_wtime();
    
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;

        #pragma omp parallel for
        for (int i = 0; i < array.size() - 1; i += 2) {
            if (array[i] > array[i + 1]) {
                swap(array[i], array[i + 1]);
                isSorted = false;
            }
        }

        #pragma omp parallel for
        for (int i = 1; i < array.size() - 1; i += 2) {
            if (array[i] > array[i + 1]) {
                swap(array[i], array[i + 1]);
                isSorted = false;
            }
        }
    }
    
    double end = omp_get_wtime();
    
    cout<<"After bubble sort odd even transposition: ";
    for(int i = 0; i < array.size(); i++)
    	cout << array[i] << " ";
    cout << endl;
    cout << "Parallel bubble sort using odd-even transposition time: " << end - start << endl;
    cout << endl;
}

void merge(int arr[], int l, int m, int h){
    
    int n1=m-l+1, n2=h-m;
    int left[n1],right[n2];
    for(int i=0;i<n1;i++)
        left[i]=arr[i+l];
    for(int j=0;j<n2;j++)
        right[j]=arr[m+1+j];    
    int i=0,j=0,k=l;
    while(i<n1 && j<n2){
        if(left[i]<=right[j])
            arr[k++]=left[i++];
        else
            arr[k++]=right[j++];
    }
    while(i<n1)
        arr[k++]=left[i++];
    while(j<n2)
        arr[k++]=right[j++];    
}

void mergesort(int array[],int low,int high){
    if(low < high){
        int mid = (low + high) / 2;
        mergesort(array,low,mid);
        mergesort(array,mid+1,high);
        merge(array,low,mid,high);
    }
}

void perform_merge_sort(int arr[],int size){
    int array[size];
    for(int i = 0 ; i < size; i++){
        array[i] = arr[i];
    }
    
    double start = omp_get_wtime();
    mergesort(array,0,size-1);
    double end = omp_get_wtime();
    
    cout << "After Merge Sort: ";
     for(int i = 0 ; i < size; i++){
         cout << array[i] << " ";
     }
    cout << endl;
    cout << "Time Required for Merge Sort: " << end - start << endl;
}

void p_mergesort(int array[],int low,int high){
    if(low < high){
        int mid = (low + high) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
                p_mergesort(array,low,mid);
            #pragma omp section
                p_mergesort(array,mid+1,high);
        }
        merge(array,low,mid,high);
    }
}

void perform_p_merge_sort(int arr[],int size){
    int array[size];
    for(int i = 0 ; i < size; i++){
        array[i] = arr[i];
    }
    
    double start = omp_get_wtime();
    p_mergesort(array,0,size-1);
    double end = omp_get_wtime();
    cout << "After Parallel Merge Sort:\n";
     for(int i = 0 ; i < size; i++){
         cout << array[i] << " ";
     }
    cout << endl;
    cout << "Time Required for Parallel Merge Sort: " << end - start << endl;
}

int main() {
    vector<int> arr;
    for (int i = 1000; i >= 0; i--) {
        arr.push_back(i);
    }
    
    cout<<"Before Bubble Sort: ";
    for(int i = 0; i < arr.size(); i++)
    	cout << arr[i] << " ";
    cout << endl;
    
    sequential_bubble_sort(arr);
    bubble_sort_odd_even(arr);
    
    int SIZE = 1000;
    int MAX = 1000;
    
    int array[SIZE];
    for(int i = 0 ; i < SIZE; i ++){
        array[i] = rand() % MAX;
    }
    
    cout<<"Before Merge Sort: ";
    for(int i = 0; i < SIZE; i++)
    	cout << array[i] << " ";
    cout << endl;
    
    perform_merge_sort(array,SIZE);
    perform_p_merge_sort(array,SIZE);
    return 0;
}

