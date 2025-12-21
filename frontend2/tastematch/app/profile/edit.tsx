import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, ScrollView, TouchableOpacity, Alert, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { getProfile, updateProfile } from '../../src/services/api';
import { Ionicons } from '@expo/vector-icons';

export default function EditProfile() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    const [bio, setBio] = useState('');
    const [city, setCity] = useState('');
    const [gender, setGender] = useState('');
    const [interestedIn, setInterestedIn] = useState('');

    useEffect(() => {
        loadProfile();
    }, []);

    const loadProfile = async () => {
        try {
            const data = await getProfile();
            setBio(data.bio || '');
            setCity(data.location_city || '');
            setGender(data.gender || '');
            setInterestedIn(data.interested_in || '');
        } catch (e) {
            Alert.alert("Error", "Could not load profile");
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            await updateProfile({
                bio,
                location_city: city,
                gender,
                interested_in: interestedIn
            });
            Alert.alert("Success", "Profile Updated!");
            router.back();
        } catch (e) {
            Alert.alert("Error", "Failed to update profile");
        } finally {
            setSaving(false);
        }
    };

    if (loading) return <View style={styles.container}><ActivityIndicator size="large" color="#FF3366" /></View>;

    return (
        <ScrollView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <Ionicons name="arrow-back" size={24} color="white" />
                </TouchableOpacity>
                <Text style={styles.title}>Edit Profile</Text>
                <View style={{ width: 24 }} />
            </View>

            <View style={styles.form}>
                <Text style={styles.label}>Bio</Text>
                <TextInput
                    style={[styles.input, { height: 100, textAlignVertical: 'top' }]}
                    value={bio}
                    onChangeText={setBio}
                    placeholder="Tell us about yourself..."
                    placeholderTextColor="#666"
                    multiline
                />

                <Text style={styles.label}>City (Location)</Text>
                <TextInput
                    style={styles.input}
                    value={city}
                    onChangeText={setCity}
                    placeholder="e.g. New York, Istanbul"
                    placeholderTextColor="#666"
                />

                <Text style={styles.label}>I am a...</Text>
                <View style={styles.row}>
                    {['male', 'female', 'other'].map(opt => (
                        <TouchableOpacity
                            key={opt}
                            style={[styles.chip, gender === opt && styles.chipActive]}
                            onPress={() => setGender(opt)}
                        >
                            <Text style={styles.chipText}>{opt.toUpperCase()}</Text>
                        </TouchableOpacity>
                    ))}
                </View>

                <Text style={styles.label}>Interested in...</Text>
                <View style={styles.row}>
                    {['male', 'female', 'both'].map(opt => (
                        <TouchableOpacity
                            key={opt}
                            style={[styles.chip, interestedIn === opt && styles.chipActive]}
                            onPress={() => setInterestedIn(opt)}
                        >
                            <Text style={styles.chipText}>{opt.toUpperCase()}</Text>
                        </TouchableOpacity>
                    ))}
                </View>

                <TouchableOpacity style={styles.saveBtn} onPress={handleSave} disabled={saving}>
                    {saving ? <ActivityIndicator color="white" /> : <Text style={styles.saveText}>Save Changes</Text>}
                </TouchableOpacity>

            </View>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#121212',
        padding: 20
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 30,
        marginTop: 20
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        color: 'white'
    },
    form: {
        gap: 20
    },
    label: {
        color: '#888',
        fontSize: 14,
        marginBottom: 8,
        marginLeft: 4
    },
    input: {
        backgroundColor: '#1E1E1E',
        borderRadius: 12,
        padding: 15,
        color: 'white',
        fontSize: 16,
        borderWidth: 1,
        borderColor: '#333'
    },
    row: {
        flexDirection: 'row',
        gap: 10
    },
    chip: {
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 20,
        backgroundColor: '#333',
    },
    chipActive: {
        backgroundColor: '#FF3366',
    },
    chipText: {
        color: 'white',
        fontWeight: '600',
        fontSize: 12
    },
    saveBtn: {
        backgroundColor: '#FF3366',
        padding: 18,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 20
    },
    saveText: {
        color: 'white',
        fontWeight: 'bold',
        fontSize: 16
    }
});
