CLASS ltcl_test_z_humaneval_162 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_hello_world FOR TESTING,
      test_empty_string FOR TESTING,
      test_a_b_c FOR TESTING,
      test_password FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_162 IMPLEMENTATION.

  METHOD test_hello_world.
    DATA: result TYPE string.

    result = z_humaneval_162=>string_to_md5( 'Hello world' ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = '3e25960a79dbc69b674cd4ec67a72c62'
    ).
  ENDMETHOD.

  METHOD test_empty_string.
    DATA: result TYPE string.

    result = z_humaneval_162=>string_to_md5( '' ).

    cl_abap_unit_assert=>assert_initial( result ).
  ENDMETHOD.

  METHOD test_a_b_c.
    DATA: result TYPE string.

    result = z_humaneval_162=>string_to_md5( 'A B C' ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = '0ef78513b0cb8cef12743f5aeb35f888'
    ).
  ENDMETHOD.

  METHOD test_password.
    DATA: result TYPE string.

    result = z_humaneval_162=>string_to_md5( 'password' ).

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = '5f4dcc3b5aa765d61d8327deb882cf99'
    ).
  ENDMETHOD.

ENDCLASS.