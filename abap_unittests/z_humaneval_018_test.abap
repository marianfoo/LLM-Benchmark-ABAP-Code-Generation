CLASS lt_test DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS:
      test_empty_string FOR TESTING,
      test_single_character_count FOR TESTING,
      test_overlapping_substring FOR TESTING,
      test_word_search FOR TESTING.

  PRIVATE SECTION.
ENDCLASS.

CLASS lt_test IMPLEMENTATION.

  METHOD test_empty_string.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_018=>how_many_times( string = '' substring = 'x' )
      exp = 0
      msg = 'Empty string test failed' ).
  ENDMETHOD.

  METHOD test_single_character_count.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_018=>how_many_times( string = 'xyxyxyx' substring = 'x' )
      exp = 4
      msg = 'Single character count test failed' ).
  ENDMETHOD.

  METHOD test_overlapping_substring.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_018=>how_many_times( string = 'cacacacac' substring = 'cac' )
      exp = 4
      msg = 'Overlapping substring test failed' ).
  ENDMETHOD.

  METHOD test_word_search.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_018=>how_many_times( string = 'john doe' substring = 'john' )
      exp = 1
      msg = 'Word search test failed' ).
  ENDMETHOD.

ENDCLASS.