CLASS z_humaneval_036_demo DEFINITION
  FINAL
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS check_50 FOR TESTING.
    METHODS check_78 FOR TESTING.
    METHODS check_79 FOR TESTING.
    METHODS check_100 FOR TESTING.
    METHODS check_200 FOR TESTING.
    METHODS check_4000 FOR TESTING.
    METHODS check_10000 FOR TESTING.
    METHODS check_100000 FOR TESTING.
ENDCLASS.

CLASS z_humaneval_036_demo IMPLEMENTATION.

  METHOD check_50.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 50 )
      exp = 0
      msg = 'Test with n=50 failed'
    ).
  ENDMETHOD.

  METHOD check_78.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 78 )
      exp = 2
      msg = 'Test with n=78 failed'
    ).
  ENDMETHOD.

  METHOD check_79.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 79 )
      exp = 3
      msg = 'Test with n=79 failed'
    ).
  ENDMETHOD.

  METHOD check_100.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 100 )
      exp = 3
      msg = 'Test with n=100 failed'
    ).
  ENDMETHOD.

  METHOD check_200.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 200 )
      exp = 6
      msg = 'Test with n=200 failed'
    ).
  ENDMETHOD.

  METHOD check_4000.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 4000 )
      exp = 192
      msg = 'Test with n=4000 failed'
    ).
  ENDMETHOD.

  METHOD check_10000.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 10000 )
      exp = 639
      msg = 'Test with n=10000 failed'
    ).
  ENDMETHOD.

  METHOD check_100000.
    DATA(lo_fizz_buzz) = NEW z_humaneval_036( ).
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_036=>fizz_buzz( 100000 )
      exp = 8026
      msg = 'Test with n=100000 failed'
    ).
  ENDMETHOD.

ENDCLASS.