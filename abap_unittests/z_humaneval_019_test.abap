CLASS lt_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS:
      test_empty_string FOR TESTING,
      test_single_word FOR TESTING,
      test_multiple_words_sorted FOR TESTING,
      test_unsorted_numbers_1 FOR TESTING,
      test_unsorted_numbers_2 FOR TESTING.

  PRIVATE SECTION.
ENDCLASS.

CLASS lt_test IMPLEMENTATION.

  METHOD test_empty_string.
    DATA(result) = z_humaneval_019=>sort_numbers( '' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '' ).
  ENDMETHOD.

  METHOD test_single_word.
    DATA(result) = z_humaneval_019=>sort_numbers( 'three' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'three' ).
  ENDMETHOD.

  METHOD test_multiple_words_sorted.
    DATA(result) = z_humaneval_019=>sort_numbers( 'three five nine' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'three five nine' ).
  ENDMETHOD.

  METHOD test_unsorted_numbers_1.
    DATA(result) = z_humaneval_019=>sort_numbers( 'five zero four seven nine eight' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'zero four five seven eight nine' ).
  ENDMETHOD.

  METHOD test_unsorted_numbers_2.
    DATA(result) = z_humaneval_019=>sort_numbers( 'six five four three two one zero' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'zero one two three four five six' ).
  ENDMETHOD.

ENDCLASS.