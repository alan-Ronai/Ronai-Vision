import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function Layout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: '#1f2937' },
          headerTintColor: '#fff',
          headerTitleStyle: { fontWeight: 'bold' },
          contentStyle: { backgroundColor: '#111827' },
          animation: 'slide_from_right'
        }}
      >
        <Stack.Screen
          name="index"
          options={{
            title: 'אפליקציית לוחם',
            headerTitleAlign: 'center'
          }}
        />
        <Stack.Screen
          name="camera"
          options={{
            title: 'צילום ושליחה',
            headerTitleAlign: 'center'
          }}
        />
      </Stack>
    </>
  );
}
