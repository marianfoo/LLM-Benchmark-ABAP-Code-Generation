CLASS z_humaneval_031_demo DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS:
      test_is_prime_6 FOR TESTING,
      test_is_prime_101 FOR TESTING,
      test_is_prime_11 FOR TESTING,
      test_is_prime_13441 FOR TESTING,
      test_is_prime_61 FOR TESTING,
      test_is_prime_4 FOR TESTING,
      test_is_prime_1 FOR TESTING,
      test_is_prime_5 FOR TESTING,
      test_is_prime_17 FOR TESTING,
      test_is_prime_5_times_17 FOR TESTING,
      test_is_prime_11_times_7 FOR TESTING,
      test_is_prime_13441_times_19 FOR TESTING.

ENDCLASS.

CLASS z_humaneval_031_demo IMPLEMENTATION.

  METHOD test_is_prime_6.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 6 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_is_prime_101.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 101 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_is_prime_11.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 11 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_is_prime_13441.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 13441 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_is_prime_61.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 61 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_is_prime_4.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 4 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_is_prime_1.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 1 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_is_prime_5.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 5 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_is_prime_17.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 17 )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_is_prime_5_times_17.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 5 * 17 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_is_prime_11_times_7.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 11 * 7 )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_is_prime_13441_times_19.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_031=>is_prime( 13441 * 19 )
      exp = abap_false ).
  ENDMETHOD.

ENDCLASS.