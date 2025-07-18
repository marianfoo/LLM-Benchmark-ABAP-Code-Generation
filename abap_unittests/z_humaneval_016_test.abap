CLASS lt_humaneval_016 DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS.

  PUBLIC SECTION.
    METHODS:
      test_empty_string FOR TESTING,
      test_simple_string FOR TESTING,
      test_concatenated_mixed_case FOR TESTING,
      test_single_distinct_character FOR TESTING,
      test_spaces_mixed_case FOR TESTING.

  PRIVATE SECTION.
ENDCLASS.

CLASS lt_humaneval_016 IMPLEMENTATION.

  METHOD test_empty_string.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_016=>count_distinct_characters( '' ).
    cl_aunit_assert=>assert_equals(
      act = lv_result
      exp = 0
      msg = 'Empty string should return 0' ).
  ENDMETHOD.

  METHOD test_simple_string.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_016=>count_distinct_characters( 'abcde' ).
    cl_aunit_assert=>assert_equals(
      act = lv_result
      exp = 5
      msg = 'abcde should return 5' ).
  ENDMETHOD.

  METHOD test_concatenated_mixed_case.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_016=>count_distinct_characters( 'abcde' && 'cade' && 'CADE' ).
    cl_aunit_assert=>assert_equals(
      act = lv_result
      exp = 5
      msg = 'Mixed case concatenated string should return 5' ).
  ENDMETHOD.

  METHOD test_single_distinct_character.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_016=>count_distinct_characters( 'aaaaAAAAaaaa' ).
    cl_aunit_assert=>assert_equals(
      act = lv_result
      exp = 1
      msg = 'All same character should return 1' ).
  ENDMETHOD.

  METHOD test_spaces_mixed_case.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_016=>count_distinct_characters( 'Jerry jERRY JeRRRY' ).
    cl_aunit_assert=>assert_equals(
      act = lv_result
      exp = 5
      msg = 'Jerry string should return 5 distinct characters' ).
  ENDMETHOD.

ENDCLASS.