CLASS lt_test DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_empty_string FOR TESTING,
      test_single_char FOR TESTING,
      test_non_palindrome FOR TESTING,
      test_already_palindrome FOR TESTING,
      test_random_word FOR TESTING.
ENDCLASS.

CLASS lt_test IMPLEMENTATION.

  METHOD test_empty_string.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_010=>make_palindrome( '' )
      exp = '' ).
  ENDMETHOD.

  METHOD test_single_char.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_010=>make_palindrome( 'x' )
      exp = 'x' ).
  ENDMETHOD.

  METHOD test_non_palindrome.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_010=>make_palindrome( 'xyz' )
      exp = 'xyzyx' ).
  ENDMETHOD.

  METHOD test_already_palindrome.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_010=>make_palindrome( 'xyx' )
      exp = 'xyx' ).
  ENDMETHOD.

  METHOD test_random_word.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_010=>make_palindrome( 'jerry' )
      exp = 'jerryrrej' ).
  ENDMETHOD.

ENDCLASS.