import { useFocusEffect, useRouter } from 'expo-router';
import React, { useContext, useState, useCallback } from 'react';
import { View, Text, StyleSheet, Button, FlatList, Image, ScrollView, RefreshControl, Modal, TouchableOpacity, Dimensions } from 'react-native';
import { AuthContext } from '../../src/context/AuthContext';
import { getProfile } from '../../src/services/api';

const { width, height } = Dimensions.get('window');

export default function ProfileScreen() {
    const router = useRouter();
    const { logout } = useContext(AuthContext);
    const [profile, setProfile] = useState<any>(null);
    const [refreshing, setRefreshing] = useState(false);
    const [selectedItem, setSelectedItem] = useState<any>(null);

    useFocusEffect(
        useCallback(() => {
            loadProfile();
        }, [])
    );

    const loadProfile = async () => {
        try {
            const data = await getProfile();
            setProfile(data);
        } catch (error) {
            console.error(error);
        }
    };

    const onRefresh = async () => {
        setRefreshing(true);
        await loadProfile();
        setRefreshing(false);
    };

    const getImageUrl = (item: any) => {
        if (item.image_url && item.image_url.startsWith('http')) return item.image_url;
        if (item.poster_path) return `https://image.tmdb.org/t/p/w200${item.poster_path}`;
        return 'https://via.placeholder.com/200x300';
    };

    const renderItem = ({ item }: { item: any }) => (
        <TouchableOpacity style={styles.item} onPress={() => setSelectedItem(item)}>
            <Image source={{ uri: getImageUrl(item) }} style={styles.image} />
            <Text numberOfLines={1} style={styles.itemTitle}>{item.title}</Text>
        </TouchableOpacity>
    );

    if (!profile) return <View style={styles.container}><Text>Loading...</Text></View>;

    return (
        <View style={{ flex: 1 }}>
            <ScrollView
                style={styles.container}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                }
            >
                <View style={styles.header}>
                    <View>
                        <Text style={styles.username}>@{profile.username}</Text>
                        {profile.location_city ? <Text style={styles.location}>📍 {profile.location_city}</Text> : null}
                    </View>
                    <View style={{ flexDirection: 'row', gap: 10 }}>
                        <Button title="Edit" onPress={() => router.push('/profile/edit')} color="#FF3366" />
                        <Button title="Logout" onPress={logout} color="red" />
                    </View>
                </View>

                {profile.bio ? <Text style={styles.bio}>{profile.bio}</Text> : null}

                <View style={styles.section}>
                    <Text style={styles.sectionTitle}>🔥 Super Likes</Text>
                    {profile.superlikes.length === 0 ? (
                        <Text style={styles.emptyText}>No super likes yet. Swipe Up!</Text>
                    ) : (
                        <FlatList
                            horizontal
                            data={profile.superlikes}
                            renderItem={renderItem}
                            keyExtractor={(item) => item.id.toString()}
                            showsHorizontalScrollIndicator={false}
                        />
                    )}
                </View>

                <View style={styles.section}>
                    <Text style={styles.sectionTitle}>👀 Watchlist</Text>
                    {profile.watchlist.length === 0 ? (
                        <Text style={styles.emptyText}>Watchlist is empty. Swipe Down!</Text>
                    ) : (
                        <FlatList
                            horizontal
                            data={profile.watchlist}
                            renderItem={renderItem}
                            keyExtractor={(item) => item.id.toString()}
                            showsHorizontalScrollIndicator={false}
                        />
                    )}
                </View>
            </ScrollView>

            {/* DETAIL MODAL */}
            <Modal
                animationType="slide"
                transparent={true}
                visible={!!selectedItem}
                onRequestClose={() => setSelectedItem(null)}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        {selectedItem && (
                            <ScrollView contentContainerStyle={styles.modalScroll}>
                                <Image
                                    source={{ uri: selectedItem.poster_path ? `https://image.tmdb.org/t/p/w500${selectedItem.poster_path}` : 'https://via.placeholder.com/500' }}
                                    style={styles.modalImage}
                                />
                                <Text style={styles.modalTitle}>{selectedItem.title}</Text>
                                <View style={styles.modalBadges}>
                                    <View style={styles.badge}><Text style={styles.badgeText}>{selectedItem.vote_average?.toFixed(1)}/10</Text></View>
                                    <View style={[styles.badge, { backgroundColor: '#444' }]}><Text style={styles.badgeText}>{selectedItem.release_date || 'Unknown'}</Text></View>
                                </View>
                                <Text style={styles.modalOverview}>{selectedItem.overview || 'No overview available.'}</Text>

                                <TouchableOpacity style={styles.closeButton} onPress={() => setSelectedItem(null)}>
                                    <Text style={styles.closeButtonText}>Close</Text>
                                </TouchableOpacity>
                            </ScrollView>
                        )}
                    </View>
                </View>
            </Modal>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#121212',
        padding: 20,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 30,
        marginTop: 40,
    },
    username: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
    location: {
        color: '#888',
        marginTop: 4
    },
    bio: {
        color: '#ccc',
        fontStyle: 'italic',
        marginBottom: 30,
        fontSize: 16
    },
    section: {
        marginBottom: 30,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 10,
        color: 'white'
    },
    emptyText: {
        color: '#888',
        fontStyle: 'italic',
    },
    item: {
        marginRight: 15,
        width: 100,
    },
    image: {
        width: 100,
        height: 150,
        borderRadius: 10,
        marginBottom: 5,
        backgroundColor: '#eee',
    },
    itemTitle: {
        fontSize: 12,
        textAlign: 'center',
    },
    // Modal Styles
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.8)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    modalContent: {
        width: width * 0.9,
        height: height * 0.8,
        backgroundColor: '#1a1a1a',
        borderRadius: 20,
        overflow: 'hidden',
    },
    modalScroll: {
        alignItems: 'center',
        padding: 20,
    },
    modalImage: {
        width: 250,
        height: 375,
        borderRadius: 15,
        marginBottom: 20,
    },
    modalTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
        textAlign: 'center',
        marginBottom: 15,
    },
    modalBadges: {
        flexDirection: 'row',
        marginBottom: 20,
    },
    badge: {
        backgroundColor: '#FFD700',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 15,
        marginHorizontal: 5,
    },
    badgeText: {
        fontWeight: 'bold',
        color: 'black',
    },
    modalOverview: {
        color: '#ccc',
        fontSize: 14,
        lineHeight: 22,
        textAlign: 'justify',
        marginBottom: 30,
    },
    closeButton: {
        backgroundColor: '#E91E63',
        paddingVertical: 12,
        paddingHorizontal: 40,
        borderRadius: 25,
        marginBottom: 20,
    },
    closeButtonText: {
        color: 'white',
        fontWeight: 'bold',
        fontSize: 16,
    }
});
